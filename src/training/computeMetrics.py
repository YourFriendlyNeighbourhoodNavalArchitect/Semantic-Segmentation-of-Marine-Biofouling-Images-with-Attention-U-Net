from torch import argmax, bincount, zeros, int64

def initializeConfusionMatrix(numClasses, device = 'cpu'):
    # Rows hold the ground truth classes, columns hold the predicted classes.
    return zeros((numClasses, numClasses), dtype = int64, device = device)

def updateConfusionMatrix(confusionMatrix, prediction, groundTruth):
    # Pixels are accumulated over the entire subset, so that metrics are not biased by the composition of each batch.
    # Logits of shape (B, C, H, W) are converted to class indices of shape (B, H, W).
    numClasses = confusionMatrix.shape[0]
    if prediction.dim() == groundTruth.dim() + 1:
        prediction = argmax(prediction, dim = 1)
    indices = groundTruth.flatten() * numClasses + prediction.flatten()
    confusionMatrix += bincount(indices, minlength = numClasses ** 2).reshape(numClasses, numClasses)
    return confusionMatrix

def computeMetrics(confusionMatrix):
    # Per-class metrics are derived from the confusion matrix, and averaged over the classes (macro average).
    confusionMatrix = confusionMatrix.double().cpu()
    TP = confusionMatrix.diag()
    FP = confusionMatrix.sum(dim = 0) - TP
    FN = confusionMatrix.sum(dim = 1) - TP
    # Classes absent from both the prediction and the ground truth are excluded from the averages.
    presentClasses = (TP + FP + FN) > 0

    def safeDivide(numerator, denominator):
        return (numerator / denominator.clamp(min = 1)).where(denominator > 0, numerator.new_zeros(()))

    perClassDice = safeDivide(2 * TP, 2 * TP + FP + FN)
    perClassIoU = safeDivide(TP, TP + FP + FN)
    perClassPrecision = safeDivide(TP, TP + FP)
    perClassRecall = safeDivide(TP, TP + FN)
    pixelAccuracy = (TP.sum() / confusionMatrix.sum().clamp(min = 1)).item()

    def macroAverage(perClass):
        return perClass[presentClasses].mean().item() if presentClasses.any() else 0.0

    # Return floating point values to facilitate further manipulation.
    return {'Dice Coefficient': macroAverage(perClassDice), 'IoU': macroAverage(perClassIoU), 'Accuracy': pixelAccuracy,
            'Precision': macroAverage(perClassPrecision), 'Recall': macroAverage(perClassRecall),
            'Per Class Dice': perClassDice.tolist(), 'Per Class IoU': perClassIoU.tolist(),
            'Per Class Precision': perClassPrecision.tolist(), 'Per Class Recall': perClassRecall.tolist()}
