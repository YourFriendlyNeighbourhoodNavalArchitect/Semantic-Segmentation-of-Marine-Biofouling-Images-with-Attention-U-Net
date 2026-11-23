import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from biofouling.annotationClasses import classRemap
from biofouling.models.architectures import DOWNSAMPLING_FACTORS

# The project is installed in editable mode (uv sync), so the project root is the parent of the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / 'configs' / 'default.toml'
SPLIT_STRATEGIES = ('stratified', 'similarity-aware')
MONITORED_METRICS = ('Loss', 'Dice Coefficient', 'IoU', 'Pixel Accuracy', 'Precision', 'Recall')

@dataclass(frozen = True)
class Paths:
    data: Path
    outputs: Path

    @property
    def images(self) -> Path:
        return self.data / 'images'

    @property
    def masks(self) -> Path:
        return self.data / 'masks'

    @property
    def labels(self) -> Path:
        # Labelbox export, one JSON object per line.
        return self.data / 'labels.ndjson'

    @property
    def metadata(self) -> Path:
        return self.data / 'metadata.json'

    @property
    def splits(self) -> Path:
        return self.data / 'splits'

    @property
    def videos(self) -> Path:
        return self.data / 'videos'

    @property
    def tracking(self) -> Path:
        # MLflow database of runs, parameters and metrics.
        return self.outputs / 'mlflow.db'

    @property
    def artifacts(self) -> Path:
        # Files attached to MLflow runs: checkpoints, exported models, plots.
        return self.outputs / 'mlruns'

    @property
    def studies(self) -> Path:
        # Optuna database of tuning studies.
        return self.outputs / 'optuna.db'

    @property
    def predictions(self) -> Path:
        return self.outputs / 'predictions'

    @property
    def visualizations(self) -> Path:
        return self.outputs / 'visualizations'

@dataclass(frozen = True)
class Split:
    strategy: str
    ratios: tuple[float, ...]

@dataclass(frozen = True)
class Model:
    architecture: str
    resolution: tuple[int, ...]

@dataclass(frozen = True)
class Training:
    batchSize: int
    numWorkers: int
    maxEpochs: int
    warmup: int
    patience: int
    plateauPatience: int
    plateauFactor: float
    learningRate: float
    monitor: str
    monitorMode: str
    deterministic: bool

@dataclass(frozen = True)
class Tuning:
    learningRates: tuple[float, ...]

@dataclass(frozen = True)
class ModelClass:
    name: str
    color: tuple[int, ...]
    annotations: tuple[str, ...]

@dataclass(frozen = True)
class Configuration:
    seed: int
    paths: Paths
    split: Split
    model: Model
    training: Training
    tuning: Tuning
    classes: tuple[ModelClass, ...]
    # Merged contents of the configuration files, for logging.
    raw: dict[str, Any]

    @property
    def classNames(self) -> list[str]:
        return [modelClass.name for modelClass in self.classes]

    @property
    def numClasses(self) -> int:
        return len(self.classes)

    @property
    def classDictionaries(self) -> list[dict[str, Any]]:
        # Plain form of the classes, as stored in checkpoints and exported models.
        return [{'name': modelClass.name, 'color': list(modelClass.color), 'annotations': list(modelClass.annotations)}
                for modelClass in self.classes]

    @property
    def classRemap(self) -> tuple[int, ...]:
        return classRemap(self.classDictionaries)

    @property
    def splitPath(self) -> Path:
        # Every strategy writes its own manifest, so that both splits can coexist.
        return self.paths.splits / f'{self.split.strategy}.json'

def mergeTables(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # Tables are merged key by key. Any other value, arrays included, is replaced as a whole.
    merged = dict(base)
    for key, value in override.items():
        merged[key] = mergeTables(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged

def buildSection(sectionClass: type, table: dict[str, Any], name: str) -> Any:
    # Unknown or missing keys are reported, so that a typo cannot silently fall back to a default.
    expected = {field.name for field in fields(sectionClass)}
    unknown, missing = set(table) - expected, expected - set(table)
    if unknown or missing:
        problems = [f'unknown keys: {", ".join(sorted(unknown))}' if unknown else '', f'missing keys: {", ".join(sorted(missing))}' if missing else '']
        raise ValueError(f'Configuration section "{name}" has ' + '; '.join(problem for problem in problems if problem) + '.')
    return sectionClass(**{key: tuple(value) if isinstance(value, list) else value for key, value in table.items()})

def validateConfiguration(configuration: Configuration) -> None:
    # Values that would otherwise only fail deep into a run.
    if configuration.split.strategy not in SPLIT_STRATEGIES:
        raise ValueError(f'Unknown split strategy "{configuration.split.strategy}". Available: {", ".join(SPLIT_STRATEGIES)}.')
    if len(configuration.split.ratios) != 3 or abs(sum(configuration.split.ratios) - 1) > 1e-9:
        raise ValueError('Split ratios must be three fractions (training, validation, testing) that sum to 1.')
    architecture, resolution = configuration.model.architecture, configuration.model.resolution
    if architecture not in DOWNSAMPLING_FACTORS:
        raise ValueError(f'Unknown architecture "{architecture}". Available: {", ".join(DOWNSAMPLING_FACTORS)}.')
    factor = DOWNSAMPLING_FACTORS[architecture]
    if len(resolution) != 2 or any(not isinstance(size, int) or size <= 0 or size % factor for size in resolution):
        raise ValueError(f'Model resolution must be two positive multiples of {factor} (height, width) for the {architecture}, '
                         f'not {list(resolution)}.')
    if configuration.training.monitor not in MONITORED_METRICS:
        raise ValueError(f'Unknown monitored metric "{configuration.training.monitor}". Available: {", ".join(MONITORED_METRICS)}.')
    if configuration.training.monitorMode not in ('max', 'min'):
        raise ValueError('Training monitorMode must be "max" or "min".')
    # Every annotation class must be absorbed by exactly one class.
    classRemap(configuration.classDictionaries)

def loadConfiguration(overridePaths: Iterable[Path] = ()) -> Configuration:
    raw = {}
    for path in [DEFAULT_CONFIGURATION, *overridePaths]:
        with open(path, 'rb') as file:
            raw = mergeTables(raw, tomllib.load(file))

    paths = Paths(**{key: PROJECT_ROOT / value for key, value in raw['paths'].items()})
    configuration = Configuration(seed = raw['seed'], paths = paths,
                                  split = buildSection(Split, raw['split'], 'split'),
                                  model = buildSection(Model, raw['model'], 'model'),
                                  training = buildSection(Training, raw['training'], 'training'),
                                  tuning = buildSection(Tuning, raw['tuning'], 'tuning'),
                                  classes = tuple(buildSection(ModelClass, table, 'classes') for table in raw['classes']),
                                  raw = raw)
    validateConfiguration(configuration)
    return configuration
