import numpy as np
from collections.abc import Sequence
from typing import Any
from matplotlib.lines import Line2D

# Helper functions shared by the dataset visualizer and the model tester.
# Classes are given as dictionaries with "name" and "color", in the order of their indices.
def calculateClassCoverage(mask: np.ndarray, classes: Sequence[dict[str, Any]]) -> dict[str, float]:
    # Calculate the percentage coverage of each class in the mask.
    mask = np.asarray(mask)
    return {modelClass['name']: np.count_nonzero(mask == index) / mask.size * 100 for index, modelClass in enumerate(classes)}

def classIndicesToRGB(mask: np.ndarray, classes: Sequence[dict[str, Any]]) -> np.ndarray:
    # Map class indices to RGB colors.
    mask = np.asarray(mask)
    RGBMask = np.zeros((*mask.shape, 3), dtype = np.uint8)
    for index, modelClass in enumerate(classes):
        RGBMask[mask == index] = modelClass['color']
    return RGBMask

def generateLegend(coverage: dict[str, float], classes: Sequence[dict[str, Any]]) -> tuple[list[str], list[Line2D]]:
    # Labels appear only for the classes which appear in the mask.
    colors = {modelClass['name']: modelClass['color'] for modelClass in classes}
    filteredCoverage = {className: classCoverage for className, classCoverage in coverage.items() if classCoverage > 0}
    legendLabels = [f'{className}: {classCoverage:.2f}%' for className, classCoverage in filteredCoverage.items()]
    handles = [Line2D([0], [0], marker = 's', color = 'w',
                      markerfacecolor = np.array(colors[className]) / 255,
                      markersize = 6) for className in filteredCoverage]
    return legendLabels, handles
