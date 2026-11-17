from torch.nn import Module, CrossEntropyLoss
from torch.nn.functional import softmax, one_hot

class LossFunction(Module):
    # Composite loss function suggested in the relevant literature [https://doi.org/10.1080/08927014.2023.2185143].
    def __init__(self, numClasses, alpha = 0.5):
        super().__init__()
        self.numClasses = numClasses
        self.alpha = alpha
        self.crossEntropyLoss = CrossEntropyLoss()

    def diceLoss(self, prediction, groundTruth):
        # Soft Dice loss, computed on class probabilities so that it remains differentiable.
        # The argmax of the prediction would block the gradient, reducing the loss to cross-entropy alone.
        groundTruth = one_hot(groundTruth, self.numClasses).permute(0, 3, 1, 2).float()
        prediction = softmax(prediction.float(), dim = 1)
        # Pixels of the whole batch are pooled, which stabilizes the loss for classes that occupy few pixels.
        intersection = (prediction * groundTruth).sum(dim = (0, 2, 3))
        union = prediction.sum(dim = (0, 2, 3)) + groundTruth.sum(dim = (0, 2, 3))
        # Smoothing term yields a perfect score for classes absent from both prediction and ground truth.
        diceScores = (2. * intersection + 1e-6) / (union + 1e-6)
        return 1 - diceScores.mean()

    def forward(self, prediction, groundTruth):
        crossEntropyLoss = self.crossEntropyLoss(prediction, groundTruth)
        diceLoss = self.diceLoss(prediction, groundTruth)
        return self.alpha * crossEntropyLoss + (1 - self.alpha) * diceLoss
