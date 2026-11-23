import json
from logging import getLogger
from matplotlib.backend_bases import KeyEvent
from matplotlib.pyplot import subplots, show
from src.configuration import Configuration
from src.data.SegmentationDataset import SegmentationDataset
from src.visualization.maskVisualization import calculateClassCoverage, classIndicesToRGB, generateLegend

logger = getLogger(__name__)

class DatasetVisualizer:
    # Browser of the annotated images, as the training subset sees them: augmented, with their masks and class coverage.
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.classes = configuration.classDictionaries
        self.dataset = self.loadDataset()
        self.currentIndex = 0
        self.figure, self.axes = subplots(1, 2)
        self.figure.subplots_adjust(left = 0.025, right = 0.975, top = 0.95, bottom = 0.025, wspace = 0.025)
        self.figure.canvas.manager.set_window_title('Dataset Visualizer')
        self.figure.canvas.mpl_connect('key_press_event', self.onKeyPress)

    def run(self) -> None:
        # The window stays open, and blocks, until it is closed.
        self.updatePlot()
        show()

    def loadDataset(self) -> SegmentationDataset:
        # All annotated images, as the training subset would see them.
        with open(self.configuration.paths.metadata, 'r', encoding = 'utf-8') as file:
            IDs = sorted(json.load(file))
        dataset = SegmentationDataset(self.configuration.paths.images, self.configuration.paths.masks, IDs, self.configuration.model.resolution,
                                      self.configuration.classRemap, augmentationFlag = True, seed = self.configuration.seed)
        logger.info(f'{len(dataset)} annotated images. Browse them with the left and right arrow keys.')
        return dataset

    def onKeyPress(self, event: KeyEvent) -> None:
        # Navigate through images with right and left arrow keys. Other keys keep their Matplotlib functions.
        if event.key in ('right', 'left'):
            self.currentIndex = (self.currentIndex + (1 if event.key == 'right' else -1)) % len(self.dataset)
            self.updatePlot()

    def updatePlot(self) -> None:
        # Utilizing Pyplot, print image and mask side by side.
        image, mask = self.dataset[self.currentIndex]
        RGBMask = classIndicesToRGB(mask.numpy(), self.classes)

        for axis in self.axes:
            axis.clear()
        self.axes[0].imshow(image.permute(1, 2, 0).numpy())
        self.axes[0].set_title(f'Image {self.dataset.IDs[self.currentIndex]}')
        self.axes[0].axis('off')
        self.axes[1].imshow(RGBMask)
        self.axes[1].set_title('Mask')
        self.axes[1].axis('off')

        # Calculate and display class coverage.
        coverage = calculateClassCoverage(mask.numpy(), self.classes)
        legendLabels, handles = generateLegend(coverage, self.classes)
        self.axes[1].legend(handles = handles, labels = legendLabels,
                            loc = 'upper right', title = 'Class Coverage',
                            title_fontsize = 10, fontsize = 8)
        self.figure.canvas.draw_idle()
