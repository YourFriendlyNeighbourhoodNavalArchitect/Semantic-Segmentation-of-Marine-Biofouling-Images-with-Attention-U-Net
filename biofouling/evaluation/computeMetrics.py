from torch import Tensor, argmax, bincount, zeros, int64

def initializeConfusionMatrix(numClasses: int, device: str = 'cpu') -> Tensor:
    # Rows hold the ground truth classes, columns hold the predicted classes.
    return zeros((numClasses, numClasses), dtype = int64, device = device)

def updateConfusionMatrix(confusionMatrix: Tensor, prediction: Tensor, groundTruth: Tensor) -> Tensor:
    # Pixels are accumulated over the entire subset, so that metrics are not biased by the composition of each batch.
    # Logits of shape (B, C, H, W) are converted to class indices of shape (B, H, W).
    numClasses = confusionMatrix.shape[0]
    if prediction.dim() == groundTruth.dim() + 1:
        prediction = argmax(prediction, dim = 1)
    indices = groundTruth.flatten() * numClasses + prediction.flatten()
    confusionMatrix += bincount(indices, minlength = numClasses ** 2).reshape(numClasses, numClasses)
    return confusionMatrix

def computeMetrics(confusionMatrix: Tensor) -> dict[str, float | list[float | None]]:
    # Per-class metrics are derived from the confusion matrix, and averaged over the classes (macro average).
    # Pixel accuracy is the fraction of correctly classified pixels, over all classes at once.
    confusionMatrix = confusionMatrix.double().cpu()
    TP = confusionMatrix.diag()
    FP = confusionMatrix.sum(dim = 0) - TP
    FN = confusionMatrix.sum(dim = 1) - TP
    # Classes absent from both the prediction and the ground truth have no defined metrics. They are excluded from the
    # averages, and reported as None.
    presentClasses = (TP + FP + FN) > 0

    def safeDivide(numerator: Tensor, denominator: Tensor) -> Tensor:
        return (numerator / denominator.clamp(min = 1)).where(denominator > 0, numerator.new_zeros(()))

    def macroAverage(perClass: Tensor) -> float:
        return perClass[presentClasses].mean().item() if presentClasses.any() else 0.0

    def perClassValues(perClass: Tensor) -> list[float | None]:
        return [value if present else None for value, present in zip(perClass.tolist(), presentClasses.tolist())]

    perClassDice = safeDivide(2 * TP, 2 * TP + FP + FN)
    perClassIoU = safeDivide(TP, TP + FP + FN)
    perClassPrecision = safeDivide(TP, TP + FP)
    perClassRecall = safeDivide(TP, TP + FN)
    pixelAccuracy = (TP.sum() / confusionMatrix.sum().clamp(min = 1)).item()

    return {'Dice Coefficient': macroAverage(perClassDice), 'IoU': macroAverage(perClassIoU), 'Pixel Accuracy': pixelAccuracy,
            'Precision': macroAverage(perClassPrecision), 'Recall': macroAverage(perClassRecall),
            'Per Class Dice': perClassValues(perClassDice), 'Per Class IoU': perClassValues(perClassIoU),
            'Per Class Precision': perClassValues(perClassPrecision), 'Per Class Recall': perClassValues(perClassRecall)}
