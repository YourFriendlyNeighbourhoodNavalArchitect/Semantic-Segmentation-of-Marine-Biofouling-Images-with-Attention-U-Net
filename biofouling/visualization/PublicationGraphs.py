import json
import numpy as np
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from textwrap import TextWrapper
from typing import Any
from matplotlib.pyplot import subplots, close
from matplotlib.colors import Normalize

logger = getLogger(__name__)

class PublicationGraphs:
    def __init__(self, metadataPath: Path, outputDirectory: Path, classes: Sequence[dict[str, Any]], classRemap: Sequence[int]):
        # Classes are given as dictionaries with "name" and "color", in the order of their indices.
        self.metadataPath = metadataPath
        self.outputDirectory = outputDirectory
        self.metadata = self.loadMetadata()
        self.totalImages = len(self.metadata)
        self.classes = classes
        self.classRemap = classRemap
        self.numClasses = len(classes)
        self.classNames = {index: modelClass['name'] for index, modelClass in enumerate(classes)}
        self.wrapper = TextWrapper(width = 10, break_long_words = False, break_on_hyphens = False)
        self.classOccurrences = np.zeros(self.numClasses, dtype = int)
        self.correlationMatrix = np.zeros((self.numClasses, self.numClasses), dtype = float)
        self.uniqueClassesCount = np.zeros(self.totalImages, dtype = int)

    def loadMetadata(self) -> dict[str, Any]:
        with open(self.metadataPath, 'r', encoding = 'utf-8') as file:
            return json.load(file)

    def analyzeMetadata(self) -> None:
        # Metadata holds annotation classes, which are mapped to the classes of the model.
        for ID, data in enumerate(self.metadata.values()):
            uniqueClasses = np.array(sorted({self.classRemap[index] for index in data['uniqueClassIndices']}), dtype = int)
            # Use broadcasting for computational efficiency.
            self.classOccurrences[uniqueClasses] += 1
            self.uniqueClassesCount[ID] = len(uniqueClasses)
            for i in uniqueClasses:
                self.correlationMatrix[i, uniqueClasses] += 1

        # Normalize the correlation matrix row-wise to calculate probabilities.
        rowSums = self.classOccurrences[:, None]
        self.correlationMatrix = np.divide(self.correlationMatrix, rowSums,
                                           out = np.zeros_like(self.correlationMatrix), where = rowSums > 0)

    def plotBarChart(self, labels: Sequence[Any], counts: Sequence[int], colors: Any, title: str) -> None:
        # Bar chart of image counts, annotated with raw counts and percentages above the bars.
        percentages = np.array(counts) / self.totalImages * 100
        figure, axes = subplots(figsize = (10, 6))
        bars = axes.bar(labels, counts, color = colors, edgecolor = 'black')
        axes.set_ylabel('Image Count', fontsize = 14)
        axes.set_title(title, fontsize = 16, fontweight = 'bold')
        axes.tick_params(axis = 'x', labelsize = 12)
        axes.tick_params(axis = 'y', labelsize = 12)
        axes.legend(handles = [axes.plot([], [])[0]], labels = [f'Dataset Size = {self.totalImages}'],
                    loc = 'upper right', fontsize = 12, handletextpad = 0, handlelength = 0, frameon = False)
        for bar, count, percentage in zip(bars, counts, percentages):
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{count} ({percentage:.1f}%)',
                      ha = 'center', va = 'bottom', fontsize = 10, color = 'black')
        figure.tight_layout()
        figure.savefig(self.outputDirectory / f'{title}.pdf', format = 'pdf')
        close(figure)

    def plotClassOccurrences(self) -> None:
        # Sort classes by occurrence frequency, in descending order.
        sortedOccurrences = sorted(enumerate(self.classOccurrences), key = lambda x: x[1], reverse = True)
        classIndices, occurrenceCounts = zip(*sortedOccurrences)
        classLabels = ['\n'.join(self.wrapper.wrap(self.classNames[index])) for index in classIndices]
        # Retrieve class-specific colors.
        colors = [tuple(i / 255 for i in self.classes[index]['color']) for index in classIndices]
        self.plotBarChart(classLabels, occurrenceCounts, colors, 'Class Occurrence Frequency')

    def plotClassCorrelation(self) -> None:
        # Plot heatmap using precomputed normalized correlation matrix.
        classLabels = ['\n'.join(self.wrapper.wrap(self.classNames[index])) for index in range(self.numClasses)]
        figure, axes = subplots(figsize = (14, 12))
        image = axes.imshow(self.correlationMatrix, cmap = 'coolwarm', norm = Normalize(0, 1))
        for i in range(self.numClasses):
            for j in range(self.numClasses):
                # Annotations contrast with the luminance of their cell.
                red, green, blue, _ = image.cmap(image.norm(self.correlationMatrix[i, j]))
                color = 'black' if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else 'white'
                axes.text(j, i, f'{self.correlationMatrix[i, j]:.2f}', ha = 'center', va = 'center', fontsize = 12, color = color)
        axes.set_xticks(range(self.numClasses), classLabels)
        axes.set_yticks(range(self.numClasses), classLabels, va = 'center')
        figure.colorbar(image, ax = axes, fraction = 0.046, pad = 0.04)
        axes.set_title('Class Correlation Matrix', fontsize = 16, fontweight = 'bold')
        axes.tick_params(axis = 'x', labelsize = 12)
        axes.tick_params(axis = 'y', labelsize = 12)
        figure.tight_layout()
        figure.savefig(self.outputDirectory / 'Class Correlation Matrix.pdf', format = 'pdf')
        close(figure)

    def plotUniqueClassesPerImage(self) -> None:
        # Count images by number of unique classes per image.
        uniqueCountDistribution = np.bincount(self.uniqueClassesCount, minlength = self.numClasses + 1)[1:]
        self.plotBarChart(range(1, self.numClasses + 1), uniqueCountDistribution, 'skyblue', 'Count of Unique Classes per Image')

    def generateGraphs(self) -> None:
        self.outputDirectory.mkdir(parents = True, exist_ok = True)
        self.analyzeMetadata()
        self.plotClassOccurrences()
        self.plotClassCorrelation()
        self.plotUniqueClassesPerImage()
        logger.info(f'Dataset statistics of {self.totalImages} images saved to {self.outputDirectory}.')
