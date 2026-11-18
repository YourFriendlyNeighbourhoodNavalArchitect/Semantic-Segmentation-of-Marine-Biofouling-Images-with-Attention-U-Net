from random import seed as PythonSeed
from numpy.random import seed as NumpySeed
from torch import manual_seed as TorchSeed, initial_seed, Generator
from torch.cuda import manual_seed_all as CUDASeed
from torch.backends import cudnn
from torch.cuda import is_available
from torch import optim
from torch.utils.data import DataLoader, get_worker_info
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from src.data.SegmentationDataset import SegmentationDataset

def seedWorker(workerID):
    # Each data loading worker receives its own seed, derived from the global seed, so that augmentations are reproducible.
    workerSeed = initial_seed() % 2**32
    PythonSeed(workerSeed)
    NumpySeed(workerSeed)
    get_worker_info().dataset.transformCompose.set_random_seed(workerSeed)

def getDataloaders(configuration, split):
    # Only the training subset is to be augmented.
    datasets = [SegmentationDataset(configuration.paths.images, configuration.paths.masks, split[subset], configuration.model.resolution,
                                    configuration.classRemap, augmentationFlag = subset == 'training', seed = configuration.seed)
                for subset in ('training', 'validation')]
    training = configuration.training
    pinMemory = is_available()
    generator = Generator()
    generator.manual_seed(configuration.seed)
    # Incomplete last batch is dropped, as very small batches distort batch normalization statistics.
    trainingDataloader = DataLoader(dataset = datasets[0], batch_size = training.batchSize, shuffle = True, drop_last = True,
                                    pin_memory = pinMemory, num_workers = training.numWorkers, persistent_workers = training.numWorkers > 0,
                                    worker_init_fn = seedWorker, generator = generator)
    # Shuffling is not required during validation.
    validationDataloader = DataLoader(dataset = datasets[1], batch_size = training.batchSize, shuffle = False,
                                      pin_memory = pinMemory, num_workers = training.numWorkers, persistent_workers = training.numWorkers > 0,
                                      worker_init_fn = seedWorker)
    return trainingDataloader, validationDataloader

def getOptimizer(parameters, learningRate, training):
    # Weight decay requires careful tuning when implemented alongside batch normalization [https://tinyurl.com/3kzm37tz].
    # For the purposes of this paper, we revert to the traditional Adam optimizer, without weight decay.
    optimizer = optim.Adam(parameters, lr = learningRate)
    # Learning rate decay routines: linear warmup, followed by decay whenever the monitored metric stops improving.
    warmup = training.warmup
    warmupScheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: (epoch + 1) / warmup if epoch < warmup else 1.0)
    mainScheduler = ReduceLROnPlateau(optimizer, mode = training.monitorMode, factor = training.plateauFactor,
                                      patience = training.plateauPatience, min_lr = 1e-6)
    return optimizer, warmupScheduler, mainScheduler

def setupDevice():
    if is_available():
        device = 'cuda'
        print('Using GPU.')
    else:
        device = 'cpu'
        print('Using CPU.')
    return device

def setSeed(seed, deterministic):
    # Set global seed to ensure reproducibility.
    PythonSeed(seed)
    NumpySeed(seed)
    TorchSeed(seed)
    CUDASeed(seed)
    # Deterministic algorithms are slower, so they are only enforced when exact reproducibility is required.
    cudnn.deterministic = deterministic
    cudnn.benchmark = not deterministic
