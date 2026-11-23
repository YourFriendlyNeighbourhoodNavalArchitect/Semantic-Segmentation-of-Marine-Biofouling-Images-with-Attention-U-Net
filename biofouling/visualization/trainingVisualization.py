from logging import getLogger
from math import log10, floor
from pathlib import Path
from typing import Any
from matplotlib.pyplot import subplots, close
from matplotlib.ticker import FuncFormatter
from optuna import Study
from optuna.visualization.matplotlib import plot_optimization_history, plot_slice

logger = getLogger(__name__)

def logResults(epoch: int, currentLR: float, trainingMetrics: dict[str, Any], validationMetrics: dict[str, Any]) -> None:
    # Key metrics of every epoch, to sanity-check training. Per-class metrics are tracked, but not printed.
    lines = [f'Epoch {epoch}, learning rate {currentLR:.3g}']
    for title, metrics in [('Training', trainingMetrics), ('Validation', validationMetrics)]:
        lines.append(f'{title:<12}' + ', '.join(f'{key} {value:.4f}' for key, value in metrics.items() if not isinstance(value, list)))
    logger.info('\n'.join(lines))

def ticksFormat(x: float, pos: int) -> str:
    # Format y-axis ticks: large values in scientific notation, small values with up to three decimals.
    if abs(x) >= 1000:
        exponent = floor(log10(abs(x)))
        mantissa = x / (10**exponent)
        mantissa = f'{mantissa:.1f}'.rstrip('0').rstrip('.')
        return r'${} \cdot 10^{{{}}}$'.format(mantissa, exponent)
    else:
        return f'{x:.3f}'.rstrip('0').rstrip('.')

def plotMetrics(history: dict[str, list[float]], bestEpoch: int, path: Path) -> Path:
    # Function to create and save plots after training ends.
    figure, axes = subplots(2, 2, figsize = (12, 12))
    figure.subplots_adjust(left = 0.075, right = 0.975, top = 0.925, bottom = 0.075, wspace = 0.15, hspace = 0.25)
    numEpochs = len(history['Training Loss'])
    epochs = range(1, numEpochs + 1)
    tickPositions = [1] + list(range(25, numEpochs + 1, 25))

    # Plot metrics, and mark the epoch whose weights are kept.
    for axis, (title, values), color in zip(axes.flatten(), history.items(), ['blue', 'green', 'red', 'magenta']):
        axis.plot(epochs, values, color = color, label = title)
        axis.axvline(bestEpoch, color = 'black', linestyle = '--', linewidth = 1)
        # Configure aesthetics.
        axis.set_xlabel('Epoch', fontsize = 14)
        axis.set_title(title, fontsize = 14, fontweight = 'bold')
        axis.grid(True)
        axis.yaxis.set_major_formatter(FuncFormatter(ticksFormat))
        axis.set_xlim(1, max(numEpochs, 2))
        axis.set_xticks(tickPositions)
        axis.set_xticklabels([str(x) for x in tickPositions], fontsize = 14)
        axis.yaxis.set_tick_params(labelsize = 14)

    # Save the final plot of a trial as a PNG.
    figure.savefig(path, dpi = 600)
    close(figure)
    return path

def plotStudy(study: Study, savePath: Path, monitor: str) -> None:
    # Monitored metric across trials, and against the learning rate of each trial.
    for name, plotFunction in [('Optimization History', plot_optimization_history), ('Learning Rate', plot_slice)]:
        axis = plotFunction(study, target_name = f'Validation {monitor}')
        figure = axis.flatten()[0].figure if hasattr(axis, 'flatten') else axis.figure
        figure.savefig(savePath / f'{name}.png', dpi = 600, bbox_inches = 'tight')
        close(figure)
