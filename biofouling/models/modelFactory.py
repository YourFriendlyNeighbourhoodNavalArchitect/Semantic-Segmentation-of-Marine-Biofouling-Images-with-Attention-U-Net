from torch.nn import Module, Conv2d, BatchNorm2d
from torch.nn.init import kaiming_normal_, constant_
from src.models.UNet import UNet
from src.models.SimpleCNN import SimpleCNN

# Attention mechanisms can be removed one at a time, to evaluate their contribution. Downsampling factors of the same
# architectures are listed in architectures.py.
ARCHITECTURES = {'Attention U-Net': lambda inChannels, numClasses: UNet(inChannels, numClasses, spatialAttention = True, channelAttention = True),
                 'Spatial Attention U-Net': lambda inChannels, numClasses: UNet(inChannels, numClasses, spatialAttention = True, channelAttention = False),
                 'U-Net': lambda inChannels, numClasses: UNet(inChannels, numClasses, spatialAttention = False, channelAttention = False),
                 'Simple CNN': lambda inChannels, numClasses: SimpleCNN(inChannels, numClasses)}

def initializeWeights(module: Module) -> None:
    # Weight initialization in neural networks improves performance and stability.
    # Refer to [https://arxiv.org/pdf/2406.00348v1].
    if isinstance(module, Conv2d):
        # Weights initialized as normally distributed, scaled by the number of output features.
        kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
        if module.bias is not None:
            constant_(module.bias, 0)

    elif isinstance(module, BatchNorm2d):
        # Identity mapping (no effect).
        constant_(module.weight, 1)
        constant_(module.bias, 0)

def initializeModel(architecture: str, numClasses: int, device: str, inChannels: int = 3) -> Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture "{architecture}". Available: {", ".join(ARCHITECTURES)}.')
    model = ARCHITECTURES[architecture](inChannels, numClasses)
    # Model shall be sent to GPU to expedite execution.
    model = model.to(device)
    model.apply(initializeWeights)
    return model
