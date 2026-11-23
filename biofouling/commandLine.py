from argparse import ArgumentParser, Namespace
from logging import Formatter, LogRecord, StreamHandler, getLogger, INFO
from pathlib import Path
from tempfile import TemporaryDirectory
from biofouling.configuration import PROJECT_ROOT, Configuration, loadConfiguration

# Modules are imported within each command, so that light commands do not load PyTorch or MLflow.

logger = getLogger(__name__)

class ConsoleFormatter(Formatter):
    # Information is printed as is; warnings and errors are prefixed with their level.
    def format(self, record: LogRecord) -> str:
        message = super().format(record)
        return message if record.levelno <= INFO else f'{record.levelname.capitalize()}: {message}'

def configureLogging() -> None:
    # Only the messages of the project are shown. Libraries keep their own settings, and report warnings only.
    handler = StreamHandler()
    handler.setFormatter(ConsoleFormatter('%(message)s'))
    projectLogger = getLogger('biofouling')
    projectLogger.addHandler(handler)
    projectLogger.setLevel(INFO)
    projectLogger.propagate = False

def convertLabels(configuration: Configuration, arguments: Namespace) -> None:
    from os import getenv
    from dotenv import load_dotenv
    from biofouling.data.Labelbox import Labelbox
    # The Labelbox API key is read from .env (see .env.example).
    load_dotenv(PROJECT_ROOT / '.env')
    Labelbox(configuration.paths.labels, configuration.paths.masks, configuration.paths.metadata, getenv('LABELBOX_API_KEY')).saveMasks()

def extractFrames(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.data.videoToFrames import videoToFrames
    videoToFrames(arguments.video, arguments.output or configuration.paths.videos, arguments.interval)

def splitDataset(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.data.SubsetSplit import SubsetSplit
    SubsetSplit(configuration.paths.metadata, configuration.splitPath, configuration.split.strategy, configuration.split.ratios,
                configuration.seed).splitDataset()

def viewDataset(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.visualization.DatasetVisualizer import DatasetVisualizer
    DatasetVisualizer(configuration).run()

def plotGraphs(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.visualization.PublicationGraphs import PublicationGraphs
    PublicationGraphs(configuration.paths.metadata, configuration.paths.visualizations, configuration.classDictionaries,
                      configuration.classRemap).generateGraphs()

def train(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.training.experimentTracking import startTracking, setExperiment
    from biofouling.training.trainingInitialization import setupDevice
    from biofouling.training.trainModel import trainModel
    from biofouling.training.exportONNX import exportRun
    startTracking(configuration)
    setExperiment(configuration)
    device = setupDevice()
    _, runID = trainModel(configuration, device, configuration.training.learningRate)
    exportRun(runID)

def tune(configuration: Configuration, arguments: Namespace) -> None:
    from mlflow import MlflowClient
    from biofouling.training.experimentTracking import startTracking, setExperiment
    from biofouling.training.trainingInitialization import setupDevice
    from biofouling.training.tuneModel import tuneModel
    from biofouling.training.exportONNX import exportRun
    startTracking(configuration)
    setExperiment(configuration)
    device = setupDevice()
    # Only the best trial is exported, unless a previous session already did so.
    bestRunID = tuneModel(configuration, device)
    if 'model.onnx' in [artifact.path for artifact in MlflowClient().list_artifacts(bestRunID)]:
        logger.info(f'Model of run {bestRunID} has already been exported.')
    else:
        exportRun(bestRunID)

def export(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.training.experimentTracking import startTracking
    from biofouling.training.exportONNX import exportRun
    startTracking(configuration)
    exportRun(arguments.run)

def test(configuration: Configuration, arguments: Namespace) -> None:
    from mlflow.artifacts import download_artifacts
    from biofouling.training.experimentTracking import startTracking, logTestResults
    from biofouling.evaluation.ModelTester import ModelTester
    startTracking(configuration)
    # The model is loaded into memory, so a downloaded copy is not needed afterwards.
    with TemporaryDirectory() as directory:
        modelPath = arguments.model or download_artifacts(run_id = arguments.run, artifact_path = 'model.onnx', dst_path = directory)
        tester = ModelTester(str(modelPath), configuration, arguments.figures)
    metrics, summaryPaths = tester.runInference()
    if tester.runID is not None:
        logTestResults(tester.runID, metrics, tester.classNames, summaryPaths)

def summarizeModel(configuration: Configuration, arguments: Namespace) -> None:
    from biofouling.visualization.visualizeModel import visualizeModel
    visualizeModel(configuration.model.architecture, configuration.numClasses, configuration.model.resolution)

def main() -> None:
    common = ArgumentParser(add_help = False)
    common.add_argument('--config', action = 'append', type = Path, default = [], metavar = 'FILE',
                        help = 'Configuration file layered over configs/default.toml. Repeatable; later files take precedence.')
    parser = ArgumentParser(prog = 'biofouling', description = 'Semantic Segmentation of Marine Biofouling with an Attention U-Net.')
    commands = parser.add_subparsers(dest = 'command', required = True, metavar = 'command')

    commands.add_parser('labels', parents = [common], help = 'Convert the Labelbox export into masks and metadata.').set_defaults(function = convertLabels)
    frames = commands.add_parser('frames', parents = [common], help = 'Extract frames from a video.')
    frames.add_argument('video', type = Path)
    frames.add_argument('--interval', type = float, default = 0.5, metavar = 'SECONDS', help = 'Time between saved frames (default: 0.5).')
    frames.add_argument('--output', type = Path, help = 'Folder of the frames (default: the videos folder of the data).')
    frames.set_defaults(function = extractFrames)
    commands.add_parser('split', parents = [common], help = 'Split the dataset into training, validation and testing subsets.').set_defaults(function = splitDataset)
    commands.add_parser('view', parents = [common], help = 'Browse augmented images and masks.').set_defaults(function = viewDataset)
    commands.add_parser('graphs', parents = [common], help = 'Plot dataset statistics for publication.').set_defaults(function = plotGraphs)
    commands.add_parser('train', parents = [common], help = 'Train with the configured learning rate, and export the model.').set_defaults(function = train)
    commands.add_parser('tune', parents = [common], help = 'Train on the grid of learning rates, and export the best model.').set_defaults(function = tune)
    exportParser = commands.add_parser('export', parents = [common], help = 'Export the checkpoint of a run in ONNX format.')
    exportParser.add_argument('--run', required = True, help = 'MLflow run ID.')
    exportParser.set_defaults(function = export)
    testParser = commands.add_parser('test', parents = [common], help = 'Evaluate an exported model on its testing subset.')
    source = testParser.add_mutually_exclusive_group(required = True)
    source.add_argument('--run', help = 'MLflow run ID, whose exported model is evaluated.')
    source.add_argument('--model', type = Path, help = 'ONNX model file.')
    testParser.add_argument('--figures', action = 'store_true', help = 'Also save a comparison figure for every image.')
    testParser.set_defaults(function = test)
    commands.add_parser('summary', parents = [common], help = 'Print the network summary.').set_defaults(function = summarizeModel)

    arguments = parser.parse_args()
    configureLogging()
    arguments.function(loadConfiguration(arguments.config), arguments)
