import numpy as np
from json import dump, loads
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch import from_numpy
from torch.utils.data import DataLoader
from matplotlib.pyplot import subplots, close
from onnxruntime import InferenceSession
from biofouling.annotationClasses import classRemap
from biofouling.data.SegmentationDataset import SegmentationDataset
from biofouling.data.SubsetSplit import loadSplit
from biofouling.training.computeMetrics import initializeConfusionMatrix, updateConfusionMatrix, computeMetrics
from biofouling.visualization.maskVisualization import calculateClassCoverage, classIndicesToRGB, generateLegend

# Images per inference call. Batching helps the CPU modestly, without the memory demands of larger batches.
BATCH_SIZE = 4

class ModelTester:
    def __init__(self, modelPath, configuration, saveFigures = False):
        # The exported model runs on CPU, as in the Windows application, so that reported metrics match the deployed numerics.
        self.session = InferenceSession(modelPath, providers = ['CPUExecutionProvider'])
        self.inputName = self.session.get_inputs()[0].name
        # Classes, resolution and split are read from the model itself, not from the current configuration.
        metadata = self.session.get_modelmeta().custom_metadata_map
        if 'classes' not in metadata:
            raise ValueError(f'{modelPath} carries no class metadata. Export it again with "biofouling export".')
        self.classes = loads(metadata['classes'])
        self.classNames = [modelClass['name'] for modelClass in self.classes]
        self.numClasses = len(self.classes)
        self.architecture = metadata['architecture']
        self.runID = metadata.get('runID')
        # Outputs are named after the run that trained the model, or else after the model file.
        self.outputDirectory = configuration.paths.predictions / (self.runID or Path(modelPath).stem)
        self.saveFigures = saveFigures
        self.dataset = self.loadDataset(configuration, metadata)
        self.loader = DataLoader(self.dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = configuration.training.numWorkers)

    def loadDataset(self, configuration, metadata):
        # The testing subset is that of the split the model was trained on. A split generated afterwards may place
        # training images in the testing subset, so it is refused.
        splitPath = configuration.paths.splits / f'{metadata["splitStrategy"]}.json'
        split, splitHash = loadSplit(splitPath)
        if splitHash != metadata['splitHash']:
            raise ValueError(f'{splitPath} has been generated again since the model was trained, so its testing subset '
                             f'may contain training images of the model.')
        dataset = SegmentationDataset(configuration.paths.images, configuration.paths.masks, split['testing'], loads(metadata['resolution']),
                                      classRemap(self.classes), augmentationFlag = False, seed = configuration.seed)
        print(f'Total number of samples in the dataset: {len(dataset)}')
        return dataset

    def predict(self, images):
        # Run the exported model on a batch of images, and convert logits to class indices.
        output = self.session.run(None, {self.inputName: images.numpy()})[0]
        return np.argmax(output, axis = 1).astype(np.uint8)

    def plotResults(self, image, prediction, groundTruth, ID, diceScore):
        # Plot the image, predicted mask, and ground truth mask.
        figure, axes = subplots(1, 3, figsize = (24, 8))
        figure.subplots_adjust(left = 0.025, right = 0.975, top = 0.95, bottom = 0.025, wspace = 0.025)
        for axis, picture, title in zip(axes, [image.permute(1, 2, 0).numpy(), classIndicesToRGB(prediction, self.classes),
                                               classIndicesToRGB(groundTruth, self.classes)],
                                        ['Image', f'Predicted Mask (Dice: {diceScore:.3f})', 'Ground Truth']):
            axis.imshow(picture)
            axis.set_title(title)
            axis.axis('off')

        for axis, mask in [(axes[1], prediction), (axes[2], groundTruth)]:
            legendLabels, handles = generateLegend(calculateClassCoverage(mask, self.classes), self.classes)
            axis.legend(handles = handles, labels = legendLabels, loc = 'upper right', title = 'Class Coverage',
                        title_fontsize = 16, fontsize = 14)

        figure.savefig(self.outputDirectory / 'Figures' / f'{ID}.png', dpi = 150, bbox_inches = 'tight')
        close(figure)

    def plotBestWorst(self, imageScores, predictions, numImages = 3):
        # Combined figure of the best and worst predictions, ranked by Dice coefficient.
        # Predictions are reused, rather than computed again.
        imageScores = sorted(imageScores, key = lambda x: x[1], reverse = True)
        figure, axes = subplots(2, 2 * numImages, figsize = (12 * numImages, 13), squeeze = False)
        figure.subplots_adjust(wspace = 0.04, hspace = 0.15)
        for row, (samples, label) in enumerate([(imageScores[:numImages], 'Best'), (imageScores[-numImages:], 'Worst')]):
            for column, (ID, diceScore) in enumerate(samples):
                image, _ = self.dataset[self.dataset.IDs.index(ID)]
                axes[row, 2 * column].imshow(image.permute(1, 2, 0).numpy())
                axes[row, 2 * column].set_title(f'Image {ID}', fontsize = 14, fontweight = 'bold')
                axes[row, 2 * column + 1].imshow(classIndicesToRGB(predictions[ID], self.classes))
                axes[row, 2 * column + 1].set_title(f'Dice: {diceScore:.3f}', fontsize = 14, fontweight = 'bold')
            axes[row, 0].text(-0.15, 0.5, f'{label} Cases', transform = axes[row, 0].transAxes, fontsize = 20, fontweight = 'bold',
                              rotation = 90, va = 'center', ha = 'center')
        for axis in axes.flatten():
            axis.axis('off')

        # Shared legend at the bottom.
        _, handles = generateLegend({className: 1 for className in self.classNames}, self.classes)
        figure.legend(handles = handles, labels = self.classNames, loc = 'lower center', ncol = self.numClasses, fontsize = 15)
        paths = [self.outputDirectory / f'Best and Worst Predictions.{extension}' for extension in ['png', 'pdf']]
        for path in paths:
            figure.savefig(path, dpi = 600, bbox_inches = 'tight')
        close(figure)
        return paths

    def plotConfusionMatrix(self, confusionMatrix):
        # Confusion matrix normalized by row, i.e. the fraction of each true class assigned to each predicted class.
        confusionMatrix = confusionMatrix.double().cpu().numpy()
        confusionMatrix = confusionMatrix / np.maximum(confusionMatrix.sum(axis = 1, keepdims = True), 1)
        figure, axes = subplots(figsize = (8, 7))
        image = axes.imshow(confusionMatrix, cmap = 'Blues', vmin = 0, vmax = 1)
        axes.set_xticks(range(self.numClasses), self.classNames, fontsize = 13, rotation = 30, ha = 'right')
        axes.set_yticks(range(self.numClasses), self.classNames, fontsize = 13)
        axes.set_xlabel('Predicted Class', fontsize = 16)
        axes.set_ylabel('True Class', fontsize = 16)
        axes.set_title('Normalized Confusion Matrix', fontsize = 18, fontweight = 'bold')
        for i in range(self.numClasses):
            for j in range(self.numClasses):
                axes.text(j, i, f'{confusionMatrix[i, j]:.2f}', ha = 'center', va = 'center', fontsize = 14, fontweight = 'bold',
                          color = 'white' if confusionMatrix[i, j] > 0.5 else 'black')
        figure.colorbar(image, ax = axes, fraction = 0.046, pad = 0.04)
        figure.tight_layout()
        paths = [self.outputDirectory / f'Confusion Matrix.{extension}' for extension in ['png', 'pdf']]
        for path in paths:
            figure.savefig(path, dpi = 600, bbox_inches = 'tight')
        close(figure)
        return paths

    def printResults(self, metrics):
        # Macro-averaged and per-class metrics, followed by a LaTeX table row of the macro-averaged metrics.
        scalarKeys = [key for key, value in metrics.items() if not isinstance(value, list)]
        print('\nMacro-averaged metrics:')
        print('\n'.join(f'{key:<20}{metrics[key]:.4f}' for key in scalarKeys))
        print('\nPer-class metrics:')
        print(f'{"":<20}' + ''.join(f'{className:>20}' for className in self.classNames))
        for key in [key for key, value in metrics.items() if isinstance(value, list)]:
            print(f'{key.replace("Per Class ", ""):<20}' + ''.join(f'{value:>20.4f}' for value in metrics[key]))
        print('\nLaTeX table row (' + ', '.join(scalarKeys) + '):')
        print(f'{self.architecture} & ' + ' & '.join(f'{metrics[key]:.4f}' for key in scalarKeys) + r' \\')

    def runInference(self):
        # Run the model on the testing subset. Metrics are accumulated over all pixels of the subset.
        # Every image is predicted once; its confusion matrix yields both its own Dice coefficient and the subset totals.
        for folder in ['Masks'] + (['Figures'] if self.saveFigures else []):
            (self.outputDirectory / folder).mkdir(parents = True, exist_ok = True)
        confusionMatrix = initializeConfusionMatrix(self.numClasses)
        imageScores = []
        predictions = {}
        IDs = iter(self.dataset.IDs)
        for images, groundTruths in tqdm(self.loader, desc = 'Testing'):
            for image, prediction, groundTruth in zip(images, self.predict(images), groundTruths):
                ID = next(IDs)
                imageMatrix = updateConfusionMatrix(initializeConfusionMatrix(self.numClasses), from_numpy(prediction).long(), groundTruth)
                confusionMatrix += imageMatrix
                # Per-image Dice coefficient, to rank predictions.
                diceScore = computeMetrics(imageMatrix)['Dice Coefficient']
                imageScores.append((ID, diceScore))
                predictions[ID] = prediction
                Image.fromarray(classIndicesToRGB(prediction, self.classes)).save(self.outputDirectory / 'Masks' / f'{ID}.png')
                if self.saveFigures:
                    self.plotResults(image, prediction, groundTruth.numpy(), ID, diceScore)

        metrics = computeMetrics(confusionMatrix)
        self.printResults(metrics)
        resultsPath = self.outputDirectory / 'testResults.json'
        with open(resultsPath, 'w') as f:
            dump({'classes': self.classNames, 'metrics': metrics, 'confusionMatrix': confusionMatrix.tolist(),
                  'imageDiceCoefficients': dict(imageScores)}, f, indent = 4)
        summaryPaths = [resultsPath, *self.plotConfusionMatrix(confusionMatrix)]
        if len(imageScores) >= 2:
            summaryPaths += self.plotBestWorst(imageScores, predictions, numImages = min(3, len(imageScores) // 2))
        print(f'Predictions saved to {self.outputDirectory}.')
        return metrics, summaryPaths
