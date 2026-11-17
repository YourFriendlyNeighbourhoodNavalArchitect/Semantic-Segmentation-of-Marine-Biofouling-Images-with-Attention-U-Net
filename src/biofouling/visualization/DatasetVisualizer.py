from json import load
from matplotlib.pyplot import subplots, draw, show
from biofouling.data.SegmentationDataset import SegmentationDataset
from biofouling.visualization.maskVisualization import calculateClassCoverage, classIndicesToRGB, generateLegend

class DatasetVisualizer:
    def __init__(self, configuration):
        self.configuration = configuration
        self.classes = configuration.classDictionaries
        self.dataset = self.loadDataset()
        self.currentIndex = 0
        self.figure, self.axes = subplots(1, 2)
        self.figure.subplots_adjust(left = 0.025, right = 0.975, top = 0.95, bottom = 0.025, wspace = 0.025)
        self.updatePlot()
        self.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)
        show()

    def loadDataset(self):
        # All annotated images, as the training subset would see them.
        with open(self.configuration.paths.metadata, 'r') as file:
            IDs = sorted(load(file))
        dataset = SegmentationDataset(self.configuration.paths.images, self.configuration.paths.masks, IDs, self.configuration.model.resolution,
                                      self.configuration.classRemap, augmentationFlag = True, seed = self.configuration.seed)
        print(f'Total number of samples in the dataset: {len(dataset)}')
        return dataset

    def onKeyPress(self, event):
        # Navigate through images with right and left arrow keys.
        if event.key == 'right':
            self.currentIndex = (self.currentIndex + 1) % len(self.dataset)
            self.updatePlot()
        elif event.key == 'left':
            self.currentIndex = (self.currentIndex - 1) % len(self.dataset)
            self.updatePlot()
        else:
            print('Invalid key pressed. Use left or right arrow keys.')

    def updatePlot(self):
        # Utilizing Pyplot, print image and mask side by side.
        image, mask = self.dataset[self.currentIndex]
        RGBMask = classIndicesToRGB(mask.numpy(), self.classes)

        self.axes[0].imshow(image.permute(1, 2, 0).numpy(), animated = True)
        self.axes[0].set_title('Image')
        self.axes[0].axis('off')
        self.axes[1].imshow(RGBMask, animated = True)
        self.axes[1].set_title('Mask')
        self.axes[1].axis('off')

        # Calculate and display class coverage.
        coverage = calculateClassCoverage(mask.numpy(), self.classes)
        legendLabels, handles = generateLegend(coverage, self.classes)
        self.axes[1].legend(handles = handles, labels = legendLabels,
                            loc = 'upper right', title = 'Class Coverage',
                            title_fontsize = 10, fontsize = 8)

        self.figure.canvas.manager.set_window_title('Dataset Visualizer')
        draw()
