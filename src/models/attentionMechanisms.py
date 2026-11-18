from torch import Tensor
from torch.nn import Module, AdaptiveAvgPool2d, Sequential, Conv2d, ReLU, Sigmoid
from torch.nn.functional import interpolate

class ChannelAttention(Module):
    # Channel attention mechanism, that adaptively recalibrates channel-wise feature responses.
    # Refer to [https://arxiv.org/abs/1709.01507v4].
    def __init__(self, channel: int):
        super().__init__()
        self.squeeze = AdaptiveAvgPool2d(1)
        self.excitation = Sequential(Conv2d(channel, channel // 16, kernel_size = 1, bias = False),
                                     ReLU(inplace = True),
                                     Conv2d(channel // 16, channel, kernel_size = 1, bias = False),
                                     Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y.expand_as(x)

class AttentionGate(Module):
    # Additive attention gate of the Attention U-Net, which suppresses irrelevant regions of the skip connection.
    # Refer to [https://arxiv.org/pdf/1804.03999], equation 1 and figure 2.
    # The gating signal g is collected from the coarser scale, before upsampling, so that it carries more context than
    # the skip connection x. Attention coefficients are computed on the grid of g, and resampled to the grid of x.
    def __init__(self, encoderChannels: int, gatingChannels: int, intermediateChannels: int):
        super().__init__()
        # W_x: the skip connection is brought to the grid of the gating signal by a strided convolution.
        # It carries no bias, as b_g is added to the same sum.
        self.encoderTransform = Conv2d(encoderChannels, intermediateChannels, kernel_size = 2, stride = 2, bias = False)
        # W_g and b_g.
        self.gatingTransform = Conv2d(gatingChannels, intermediateChannels, kernel_size = 1)
        # psi and b_psi: one attention coefficient per position.
        self.attentionTransform = Conv2d(intermediateChannels, 1, kernel_size = 1)
        self.relu = ReLU(inplace = True)
        self.sigmoid = Sigmoid()

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        additiveAttention = self.relu(self.encoderTransform(x) + self.gatingTransform(g))
        attentionCoefficients = self.sigmoid(self.attentionTransform(additiveAttention))
        # Grid resampling of the attention coefficients, by bilinear interpolation.
        attentionCoefficients = interpolate(attentionCoefficients, size = x.shape[2:], mode = 'bilinear', align_corners = False)
        # Regions with high/low attention coefficients are promoted/suppressed.
        return x * attentionCoefficients
