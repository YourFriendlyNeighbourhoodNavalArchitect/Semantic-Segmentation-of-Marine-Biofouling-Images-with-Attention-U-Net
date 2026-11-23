import json
import mlflow
from logging import getLogger
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any
from optuna import Study, create_study
from optuna.trial import FrozenTrial, Trial, TrialState
from optuna.samplers import RandomSampler
from optuna.pruners import MedianPruner
from mlflow import MlflowClient
from biofouling.configuration import Configuration
from biofouling.training.trainModel import trainModel
from biofouling.training.experimentTracking import experimentName, flattenTables
from biofouling.visualization.trainingVisualization import plotStudy

logger = getLogger(__name__)

# Settings that do not affect the trials of a study: the learning rate is replaced by the grid, and the number of data
# loading workers changes the speed only, as augmentations are seeded per worker.
INDEPENDENT_TRAINING_SETTINGS = ('learningRate', 'numWorkers')

def studyConfiguration(configuration: Configuration) -> dict[str, Any]:
    # Settings that a resumed study must share with the study it continues. The tuned grid itself may be extended.
    settings = {key: value for key, value in configuration.raw.items() if key not in ('paths', 'tuning')}
    settings['training'] = {key: value for key, value in settings['training'].items() if key not in INDEPENDENT_TRAINING_SETTINGS}
    return settings

def checkStudyConfiguration(study: Study, configuration: Configuration) -> None:
    # A resumed study must not mix trials of different settings.
    current = studyConfiguration(configuration)
    stored = study.user_attrs.get('configuration')
    if stored is None:
        study.set_user_attr('configuration', json.dumps(current))
        return
    # Settings are compared one by one, so that the message names each setting that differs, e.g. "training.patience".
    stored, current = flattenTables(json.loads(stored)), flattenTables(current)
    differences = sorted(key for key in stored.keys() | current.keys() if stored.get(key) != current.get(key))
    if differences:
        raise ValueError(f'Study "{study.study_name}" was started with different settings ({", ".join(differences)}), '
                         f'so its trials are not comparable with new ones. Restore the settings, or delete the study from '
                         f'{configuration.paths.studies}.')

def trialLearningRate(trial: FrozenTrial) -> float | None:
    # Learning rate of a trial. Queued trials that have not started yet hold it among their fixed parameters.
    return trial.params.get('learningRate', trial.system_attrs.get('fixed_params', {}).get('learningRate'))

def queueRemainingPoints(study: Study, learningRates: list[float]) -> int:
    # A study is run by a single process, so trials still marked as running were interrupted (e.g. the machine was shut down).
    client = MlflowClient()
    for trial in study.get_trials(deepcopy = False, states = (TrialState.RUNNING,)):
        study.tell(trial.number, state = TrialState.FAIL)
        if 'runID' in trial.user_attrs:
            client.set_terminated(trial.user_attrs['runID'], 'KILLED')

    # Every point of the grid that has not been completed is queued once, interrupted points included.
    def learningRatesOf(*states: TrialState) -> set[float | None]:
        return {trialLearningRate(trial) for trial in study.get_trials(deepcopy = False, states = states)}

    completed = learningRatesOf(TrialState.COMPLETE, TrialState.PRUNED)
    waiting = learningRatesOf(TrialState.WAITING)
    interrupted = learningRatesOf(TrialState.FAIL) - completed
    for learningRate in sorted(interrupted & set(learningRates)):
        logger.info(f'Learning rate {learningRate:g} was interrupted, and is trained again.')
    for learningRate in learningRates:
        if learningRate not in completed and learningRate not in waiting:
            study.enqueue_trial({'learningRate': learningRate})
    return len(study.get_trials(deepcopy = False, states = (TrialState.WAITING,)))

def tuneModel(configuration: Configuration, device: str) -> str:
    # Learning rates are evaluated on a fixed grid. Every point is trained once: a resumed or extended study only runs
    # the points that have not been completed, so that no training is repeated.
    learningRates = sorted(configuration.tuning.learningRates)
    monitor = configuration.training.monitor
    study = create_study(study_name = experimentName(configuration), storage = f'sqlite:///{configuration.paths.studies.as_posix()}',
                         load_if_exists = True, direction = 'maximize' if configuration.training.monitorMode == 'max' else 'minimize',
                         # Trials are queued with fixed learning rates, so the sampler is never consulted. Optuna's grid
                         # sampler is avoided on purpose: it repeats completed points once its grid is exhausted or extended.
                         sampler = RandomSampler(seed = configuration.seed),
                         # Unpromising trials are stopped early, once a few trials have completed.
                         pruner = MedianPruner(n_startup_trials = 4, n_warmup_steps = 30))
    checkStudyConfiguration(study, configuration)
    numRemaining = queueRemainingPoints(study, learningRates)

    def objective(trial: Trial) -> float:
        learningRate = trial.suggest_float('learningRate', learningRates[0], learningRates[-1], log = True)
        validationMetrics, _ = trainModel(configuration, device, learningRate, trial)
        return validationMetrics[monitor]

    # Trials are nested under one parent run per study, which a resumed study continues.
    parentRunID = study.user_attrs.get('runID')
    with mlflow.start_run(run_id = parentRunID, run_name = None if parentRunID else 'Tuning') as parentRun:
        if parentRunID is None:
            study.set_user_attr('runID', parentRun.info.run_id)
        # The grid may be extended between sessions, so it is recorded as a tag, which can be overwritten.
        mlflow.set_tag('tuning.learningRates', json.dumps(learningRates))
        if numRemaining:
            study.optimize(objective, n_trials = numRemaining)
        else:
            logger.info('Every learning rate of the grid has been trained.')

        completedTrials = study.get_trials(deepcopy = False, states = (TrialState.COMPLETE,))
        if not completedTrials:
            raise RuntimeError('No trial of the study has completed.')
        bestTrial = study.best_trial
        with TemporaryDirectory() as directory:
            plotStudy(study, Path(directory), monitor)
            mlflow.log_artifacts(directory)
        mlflow.log_metric(f'best/validation/{monitor}', bestTrial.value)
        mlflow.set_tag('bestRunID', bestTrial.user_attrs['runID'])

    logger.info(f'Best trial {bestTrial.number}: learning rate {bestTrial.params["learningRate"]:g}, {monitor} = {bestTrial.value:.4f}.')
    return bestTrial.user_attrs['runID']
