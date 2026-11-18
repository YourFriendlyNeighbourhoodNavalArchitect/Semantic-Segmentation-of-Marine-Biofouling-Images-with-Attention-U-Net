from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, MaxPool2d
from torch.nn.functional import relu, interpolate

class SimpleCNN(Module):
    # Simple architecture utilizing successive convolution blocks.
    # Missing skip connections and encoder-decoder structure.
    # Serves purpose of comparison with Attention U-Net to evaluate accuracy results.
    def __init__(self, inChannels: int, numClasses: int):
        super().__init__()
        self.numClasses = numClasses

        # Convolutions followed by batch normalization carry no bias, which the normalization would cancel.
        self.convolutionOne = Conv2d(inChannels, 64, kernel_size = 3, padding = 1, bias = False)
        self.batchNormOne = BatchNorm2d(64)

        self.convolutionTwo = Conv2d(64, 128, kernel_size = 3, padding = 1, bias = False)
        self.batchNormTwo = BatchNorm2d(128)

        self.convolutionThree = Conv2d(128, 256, kernel_size = 3, padding = 1, bias = False)
        self.batchNormThree = BatchNorm2d(256)

        self.fullyConnectedOne = Conv2d(256, 128, kernel_size = 1, bias = False)
        self.batchNormFour = BatchNorm2d(128)

        self.fullyConnectedTwo = Conv2d(128, numClasses, kernel_size = 1)

        self.pooling = MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.batchNormOne(self.convolutionOne(x)))
        x = self.pooling(x)

        x = relu(self.batchNormTwo(self.convolutionTwo(x)))
        x = self.pooling(x)

        x = relu(self.batchNormThree(self.convolutionThree(x)))
        x = self.pooling(x)

        x = relu(self.batchNormFour(self.fullyConnectedOne(x)))
        x = self.fullyConnectedTwo(x)

        x = interpolate(x, scale_factor = 8, mode = 'bilinear', align_corners = False)
        return x
