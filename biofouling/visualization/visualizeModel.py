from collections.abc import Sequence
from torchinfo import summary
from biofouling.models.modelFactory import initializeModel
from biofouling.training.trainingInitialization import setupDevice

def visualizeModel(modelName: str, numClasses: int, resolution: Sequence[int]) -> None:
    # Layer-by-layer summary of the network: output shapes, parameters and multiply-adds, for one image of the input resolution.
    device = setupDevice()
    model = initializeModel(modelName, numClasses = numClasses, device = device)
    summary(model, input_size = (1, 3, *resolution), device = device, depth = 4)
