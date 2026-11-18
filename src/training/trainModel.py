import mlflow
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from torch.cuda import max_memory_allocated
from optuna import Trial
from optuna.exceptions import TrialPruned
from src.configuration import Configuration
from src.data.SubsetSplit import loadSplit
from src.models.modelFactory import initializeModel
from src.training.LossFunction import LossFunction
from src.training.trainingInitialization import getDataloaders, getOptimizer, setSeed
from src.training.trainingLoop import trainingLoop
from src.training.checkpoints import saveCheckpoint
from src.training.experimentTracking import logRunDetails, logEpoch, logBestEpoch
from src.visualization.trainingVisualization import plotMetrics

logger = getLogger(__name__)

def trainModel(configuration: Configuration, device: str, learningRate: float, trial: Trial | None = None) -> tuple[dict[str, Any], str]:
    # Every training run starts from the same seed, so that trials differ only in their hyperparameters,
    # and any single trial can be reproduced on its own.
    setSeed(configuration.seed, configuration.training.deterministic)
    split, splitHash = loadSplit(configuration.splitPath)
    runName = f'Learning Rate {learningRate:g}' if trial is None else f'Trial {trial.number}, Learning Rate {learningRate:g}'
    # The run is ended explicitly, so that a pruned trial is recorded as such, rather than as a failure.
    run = mlflow.start_run(run_name = runName, nested = trial is not None)
    status = 'FAILED'
    try:
        logRunDetails(configuration, learningRate, splitHash, device)
        mlflow.log_artifact(str(configuration.splitPath))
        if trial is not None:
            mlflow.set_tag('trial', trial.number)
            trial.set_user_attr('runID', run.info.run_id)

        classNames = configuration.classNames
        criterion = LossFunction(configuration.numClasses)
        model = initializeModel(configuration.model.architecture, configuration.numClasses, device)
        optimizer, warmupScheduler, mainScheduler = getOptimizer(model.parameters(), learningRate, configuration.training)
        trainingDataloader, validationDataloader = getDataloaders(configuration, split)
        trainingMetrics, validationMetrics, bestEpoch, history = trainingLoop(
            model, trainingDataloader, validationDataloader, optimizer, warmupScheduler, mainScheduler, criterion, configuration.numClasses,
            device, configuration.training, onEpoch = lambda *metrics: logEpoch(*metrics, classNames), trial = trial)

        # Checkpoint of the best epoch and training plot are attached to the run.
        with TemporaryDirectory() as directory:
            saveCheckpoint(Path(directory) / 'checkpoint.pt', model, configuration.model.architecture, configuration.classDictionaries,
                           configuration.model.resolution, bestEpoch, configuration.split.strategy, splitHash)
            plotMetrics(history, bestEpoch, Path(directory) / 'trainingPlot.png')
            mlflow.log_artifacts(directory)
        logBestEpoch(bestEpoch, trainingMetrics, validationMetrics, classNames)
        if device == 'cuda':
            mlflow.log_metric('peakMemoryGiB', max_memory_allocated() / 2**30)
        status = 'FINISHED'
        logger.info(f'Run {run.info.run_id} finished.')
        return validationMetrics, run.info.run_id
    except TrialPruned:
        status = 'KILLED'
        mlflow.set_tag('pruned', 'true')
        raise
    except KeyboardInterrupt:
        status = 'KILLED'
        raise
    finally:
        mlflow.end_run(status)
