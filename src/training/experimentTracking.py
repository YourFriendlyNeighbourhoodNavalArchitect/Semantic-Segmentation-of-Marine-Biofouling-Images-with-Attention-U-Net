from json import dumps
from subprocess import run, DEVNULL, CalledProcessError
from importlib.metadata import version
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from src.configuration import PROJECT_ROOT

# Versions of the packages that determine the numerical results of a run.
TRACKED_PACKAGES = ('torch', 'numpy', 'albumentations', 'albucore', 'opencv-python-headless', 'optuna', 'onnx', 'onnxruntime')

def startTracking(configuration):
    # Runs, parameters and metrics are stored in an SQLite database; attached files are stored in a folder next to it.
    configuration.paths.outputs.mkdir(parents = True, exist_ok = True)
    mlflow.set_tracking_uri(f'sqlite:///{configuration.paths.tracking.as_posix()}')

def experimentName(configuration):
    # Runs are grouped by everything that makes their results incomparable.
    return f'{configuration.model.architecture}, {configuration.numClasses} Classes, {configuration.split.strategy.capitalize()} Split'

def setExperiment(configuration):
    name = experimentName(configuration)
    client = MlflowClient()
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location = configuration.paths.artifacts.as_uri())
    mlflow.set_experiment(name)
    return name

def flattenTables(table, prefix = ''):
    # Nested configuration tables are flattened into dotted parameter names, e.g. "training.batchSize".
    flat = {}
    for key, value in table.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(flattenTables(value, f'{name}.'))
        else:
            flat[name] = dumps(value) if isinstance(value, list) else value
    return flat

def gitCommit():
    # Commit of the code that produced a run, marked as dirty if there are uncommitted changes.
    try:
        commit = run(['git', 'rev-parse', 'HEAD'], cwd = PROJECT_ROOT, capture_output = True, text = True, check = True).stdout.strip()
        dirty = run(['git', 'diff', '--quiet', 'HEAD'], cwd = PROJECT_ROOT, stdout = DEVNULL, stderr = DEVNULL).returncode != 0
        return f'{commit}-dirty' if dirty else commit
    except (OSError, CalledProcessError):
        return 'unknown'

def logRunDetails(configuration, learningRate, splitHash, device):
    # Everything needed to trace and reproduce a run: configuration, split, code and package versions, hardware.
    parameters = flattenTables({key: value for key, value in configuration.raw.items() if key not in ('paths', 'tuning')})
    parameters['training.learningRate'] = learningRate
    parameters['splitHash'] = splitHash
    mlflow.log_params(parameters)
    mlflow.log_dict(configuration.raw, 'configuration.json')
    tags = {'gitCommit': gitCommit(), 'device': device}
    tags.update({f'version.{package}': version(package) for package in TRACKED_PACKAGES})
    if device == 'cuda':
        from torch.cuda import get_device_name
        tags['gpu'] = get_device_name()
    mlflow.set_tags(tags)

def metricsToKeys(metrics, classNames, prefix):
    # Scalar metrics are named "<prefix>/<metric>", per-class metrics "<prefix>/<metric>/<class>".
    keys = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            metricName = key.removeprefix('Per Class ')
            keys.update({f'{prefix}/{metricName}/{className}': classValue for className, classValue in zip(classNames, value)})
        else:
            keys[f'{prefix}/{key}'] = value
    return keys

def logEpoch(epoch, currentLR, trainingMetrics, validationMetrics, classNames):
    metrics = {'learningRate': currentLR}
    metrics.update(metricsToKeys(trainingMetrics, classNames, 'training'))
    metrics.update(metricsToKeys(validationMetrics, classNames, 'validation'))
    mlflow.log_metrics(metrics, step = epoch)

def logBestEpoch(bestEpoch, trainingMetrics, validationMetrics, classNames):
    # Metrics of the epoch whose weights are kept, for comparison across runs.
    metrics = {'bestEpoch': bestEpoch}
    metrics.update(metricsToKeys(trainingMetrics, classNames, 'best/training'))
    metrics.update(metricsToKeys(validationMetrics, classNames, 'best/validation'))
    mlflow.log_metrics(metrics)

def logTestResults(runID, metrics, classNames, artifactPaths):
    # Test results are attached to the run that trained the model, if it is part of the tracking database.
    client = MlflowClient()
    try:
        client.get_run(runID)
    except MlflowException:
        print(f'Run {runID} is not part of the tracking database. Test results are only saved locally.')
        return
    for key, value in metricsToKeys(metrics, classNames, 'test').items():
        client.log_metric(runID, key, value)
    for path in artifactPaths:
        client.log_artifact(runID, str(path), 'test')
    print(f'Test results attached to run {runID}.')
