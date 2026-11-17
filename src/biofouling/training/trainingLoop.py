from copy import deepcopy
from tqdm import tqdm
from torch import no_grad, autocast
from torch.amp import GradScaler
from optuna.exceptions import TrialPruned
from biofouling.training.computeMetrics import initializeConfusionMatrix, updateConfusionMatrix, computeMetrics
from biofouling.visualization.trainingVisualization import logResults

def trainOneEpoch(model, trainingDataloader, optimizer, scaler, criterion, numClasses, device):
    model.train()
    totalLoss = 0.0
    confusionMatrix = initializeConfusionMatrix(numClasses, device)

    for data in tqdm(trainingDataloader, desc = 'Training'):
        # Send data to GPU.
        image = data[0].to(device, non_blocking = True)
        groundTruth = data[1].to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)
        # Input tensor form: (B, C, H, W)
        # Ground truth tensor form: (B, H, W)
        # Mixed precision accelerates training on GPU, with negligible effect on accuracy.
        with autocast(device_type = device, enabled = device == 'cuda'):
            prediction = model(image)
            loss = criterion(prediction, groundTruth)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Metrics are accumulated without synchronizing with the GPU after every batch.
        totalLoss += loss.detach()
        updateConfusionMatrix(confusionMatrix, prediction.detach(), groundTruth)

    metrics = {'Loss': (totalLoss / len(trainingDataloader)).item()}
    metrics.update(computeMetrics(confusionMatrix))
    return metrics

def validateOneEpoch(model, validationDataloader, criterion, numClasses, device):
    model.eval()
    totalLoss = 0.0
    confusionMatrix = initializeConfusionMatrix(numClasses, device)

    with no_grad():
        for data in tqdm(validationDataloader, desc = 'Validation'):
            image = data[0].to(device, non_blocking = True)
            groundTruth = data[1].to(device, non_blocking = True)
            with autocast(device_type = device, enabled = device == 'cuda'):
                prediction = model(image)
                loss = criterion(prediction, groundTruth)
            totalLoss += loss * len(image)
            updateConfusionMatrix(confusionMatrix, prediction, groundTruth)

    # Loss is weighted by batch size, as the last batch may be smaller.
    metrics = {'Loss': (totalLoss / len(validationDataloader.dataset)).item()}
    metrics.update(computeMetrics(confusionMatrix))
    return metrics

def trainingLoop(model, trainingDataloader, validationDataloader, optimizer, warmupScheduler, mainScheduler, criterion, numClasses,
                 device, training, onEpoch, trial = None):
    # The training section of the configuration holds the stopping and monitoring settings.
    # onEpoch receives the metrics of every epoch, for logging.
    scaler = GradScaler(device, enabled = device == 'cuda')
    history = {'Training Loss': [], 'Validation Loss': [], 'Dice Coefficient': [], 'IoU': []}
    monitor = training.monitor
    # Early stopping mechanism.
    bestValue = float('-inf') if training.monitorMode == 'max' else float('inf')
    bestState = None
    bestEpoch = 0
    patienceCounter = 0
    epoch = 0

    def isImprovement(value):
        return value > bestValue if training.monitorMode == 'max' else value < bestValue

    while epoch < training.maxEpochs:
        trainingMetrics = trainOneEpoch(model, trainingDataloader, optimizer, scaler, criterion, numClasses, device)
        validationMetrics = validateOneEpoch(model, validationDataloader, criterion, numClasses, device)
        currentLR = optimizer.param_groups[0]['lr']
        epoch += 1
        if epoch < training.warmup:
            warmupScheduler.step()
        else:
            mainScheduler.step(validationMetrics[monitor])

        # Logging of metrics.
        history['Training Loss'].append(trainingMetrics['Loss'])
        history['Validation Loss'].append(validationMetrics['Loss'])
        history['Dice Coefficient'].append(validationMetrics['Dice Coefficient'])
        history['IoU'].append(validationMetrics['IoU'])
        logResults(epoch, currentLR, trainingMetrics, validationMetrics)
        onEpoch(epoch, currentLR, trainingMetrics, validationMetrics)

        if trial is not None:
            trial.report(validationMetrics[monitor], epoch)
            if trial.should_prune():
                # Get rid of unpromising trials early, to save on computational resources.
                raise TrialPruned()

        # Models train until the monitored metric stops improving. Weights of the best epoch are kept.
        if isImprovement(validationMetrics[monitor]):
            bestValue = validationMetrics[monitor]
            bestState = deepcopy(model.state_dict())
            bestEpoch = epoch
            bestMetrics = (trainingMetrics, validationMetrics)
            patienceCounter = 0
        else:
            patienceCounter += 1
        if patienceCounter >= training.patience:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

    # Restore the weights of the best epoch, rather than those of the last epoch.
    model.load_state_dict(bestState)
    print(f'Best {monitor} of {bestValue:.4f} reached at epoch {bestEpoch}.')
    trainingMetrics, validationMetrics = bestMetrics
    return trainingMetrics, validationMetrics, bestEpoch, history
