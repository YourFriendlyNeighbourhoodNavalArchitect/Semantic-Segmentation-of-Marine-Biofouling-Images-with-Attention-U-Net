from torch.nn import Module, AdaptiveAvgPool2d, Sequential, Conv2d, BatchNorm2d, ReLU, Sigmoid

class ChannelAttention(Module):
    # Channel attention mechanism, that adaptively recalibrates channel-wise feature responses.
    # Refer to [https://arxiv.org/abs/1709.01507v4].
    def __init__(self, channel):
        super().__init__()
        self.squeeze = AdaptiveAvgPool2d(1)
        self.excitation = Sequential(Conv2d(channel, channel // 16, kernel_size = 1, bias = False),
                                     ReLU(inplace = True),
                                     Conv2d(channel // 16, channel, kernel_size = 1, bias = False),
                                     Sigmoid())

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)

class SpatialAttention(Module):
    # Spatial attention mechanism, that allows the network to focus on specific regions of the input image.
    # Refer to [https://arxiv.org/pdf/1804.03999].
    # This mechanism was not implemented in the original U-Net architecture, but may offer performance improvements.
    def __init__(self, decoderInput, encoderInput, intermediateChannels):
        super().__init__()
        # Sub-module that processes the decoder input. Kernel size of 1 does not affect spatial dimensions.
        self.decoderModule = Sequential(Conv2d(decoderInput, intermediateChannels, kernel_size = 1),
                                        BatchNorm2d(intermediateChannels))
        # Sub-module that processes the encoder input.
        self.encoderModule = Sequential(Conv2d(encoderInput, intermediateChannels, kernel_size = 1),
                                        BatchNorm2d(intermediateChannels))
        # Attention mechanism, that calculates the required attention coefficients.
        self.attentionModule = Sequential(Conv2d(intermediateChannels, 1, kernel_size = 1),
                                          BatchNorm2d(1),
                                          Sigmoid())
        self.relu = ReLU(inplace = True)

    def forward(self, g, x):
        g1 = self.decoderModule(g)
        x1 = self.encoderModule(x)
        attentionCoefficients = self.relu(g1 + x1)
        attentionCoefficients = self.attentionModule(attentionCoefficients)
        # Regions with high/low attention coefficients are promoted/suppressed.
        return x * attentionCoefficients
