from torch import Tensor, cat
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, Identity, MaxPool2d, Upsample
from src.models.attentionMechanisms import ChannelAttention, AttentionGate

class BlueArrow(Module):
    # Fundamental component of the original U-Net.
    # Refer to [https://arxiv.org/pdf/1505.04597].
    def __init__(self, inChannels: int, outChannels: int, channelAttention: bool = True):
        super().__init__()
        # Convolutions followed by batch normalization carry no bias, which the normalization would cancel.
        self.convolutionOperation = Sequential(Conv2d(inChannels, outChannels, kernel_size = 3, padding = 1, bias = False),
                                               # Batch normalization is not included in the original paper.
                                               # Authors included it in their subsequent work [https://arxiv.org/pdf/1606.06650].
                                               # Here it has thus been implemented.
                                               BatchNorm2d(outChannels),
                                               ReLU(inplace = True),
                                               Conv2d(outChannels, outChannels, kernel_size = 3, padding = 1, bias = False),
                                               BatchNorm2d(outChannels),
                                               ReLU(inplace = True),
                                               # Channel attention recalibrates the features of the block.
                                               ChannelAttention(outChannels) if channelAttention else Identity())

    def forward(self, x: Tensor) -> Tensor:
        return self.convolutionOperation(x)

class DownSample(Module):
    # Building block of the contractive path of the network.
    def __init__(self, inChannels: int, outChannels: int, channelAttention: bool = True):
        super().__init__()
        self.convolution = BlueArrow(inChannels, outChannels, channelAttention)
        self.redArrow = MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        down = self.convolution(x)
        pooling = self.redArrow(down)
        return down, pooling

class UpSample(Module):
    # Building block of the expansive path of the network.
    def __init__(self, inChannels: int, outChannels: int, spatialAttention: bool = True, channelAttention: bool = True):
        super().__init__()
        # Modern decoder architectures use upsampling blocks, instead of transpose convolutions.
        self.greenArrow = Sequential(Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
                                     Conv2d(inChannels, inChannels // 2, kernel_size = 3, padding = 1, bias = False),
                                     BatchNorm2d(inChannels // 2),
                                     ReLU(inplace = True))
        # The attention gate weighs the skip connection with the coarser decoder features, before they are upsampled.
        # Intermediate channels are chosen for computational efficiency and dimensionality reduction.
        # Without it, skip connections pass the encoder features unaltered, as in the original U-Net.
        self.attentionGate = AttentionGate(encoderChannels = inChannels // 2, gatingChannels = inChannels,
                                           intermediateChannels = inChannels // 4) if spatialAttention else None
        self.convolution = BlueArrow(inChannels, outChannels, channelAttention)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self.attentionGate is not None:
            x2 = self.attentionGate(x1, x2)
        x1 = self.greenArrow(x1)
        x = cat([x1, x2], dim = 1)
        return self.convolution(x)

class UNet(Module):
    # Architecture largely based on the original paper.
    # Network output is of the form (B, C, H, W).
    # Spatial attention gates the skip connections [https://arxiv.org/pdf/1804.03999].
    # Channel attention recalibrates every convolution block [https://arxiv.org/abs/1709.01507v4].
    # Without either, the network reduces to the original U-Net, which serves as a baseline.
    def __init__(self, inChannels: int, numClasses: int, spatialAttention: bool = True, channelAttention: bool = True):
        super().__init__()
        self.numClasses = numClasses

        self.downConvolutionOne = DownSample(inChannels, 64, channelAttention)
        self.downConvolutionTwo = DownSample(64, 128, channelAttention)
        self.downConvolutionThree = DownSample(128, 256, channelAttention)
        self.downConvolutionFour = DownSample(256, 512, channelAttention)

        self.bottleneck = BlueArrow(512, 1024, channelAttention)

        self.upConvolutionOne = UpSample(1024, 512, spatialAttention, channelAttention)
        self.upConvolutionTwo = UpSample(512, 256, spatialAttention, channelAttention)
        self.upConvolutionThree = UpSample(256, 128, spatialAttention, channelAttention)
        self.upConvolutionFour = UpSample(128, 64, spatialAttention, channelAttention)

        self.output = Conv2d(64, numClasses, kernel_size = 1)

    def forward(self, x: Tensor) -> Tensor:
        downOne, poolingOne = self.downConvolutionOne(x)
        downTwo, poolingTwo = self.downConvolutionTwo(poolingOne)
        downThree, poolingThree = self.downConvolutionThree(poolingTwo)
        downFour, poolingFour = self.downConvolutionFour(poolingThree)

        bottleneck = self.bottleneck(poolingFour)

        upOne = self.upConvolutionOne(bottleneck, downFour)
        upTwo = self.upConvolutionTwo(upOne, downThree)
        upThree = self.upConvolutionThree(upTwo, downTwo)
        upFour = self.upConvolutionFour(upThree, downOne)

        output = self.output(upFour)
        return output
