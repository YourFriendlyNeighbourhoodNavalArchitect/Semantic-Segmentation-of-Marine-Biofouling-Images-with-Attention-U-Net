from torch import randn
from torchviz import make_dot
from torchinfo import summary
from biofouling.models.modelFactory import initializeModel
from biofouling.training.trainingInitialization import setupDevice

def visualizeModel(modelName, numClasses, resolution, outputDirectory):
    # Dummy script to visualize the network and confirm its structure and output form.
    device = setupDevice()
    # Segmentation is performed amongst the classes of the configuration.
    model = initializeModel(modelName, numClasses = numClasses, device = device)
    summary(model, input_size = (1, 3, *resolution), device = device)

    dummyInput = randn(1, 3, *resolution).to(device)
    output = model(dummyInput)
    print(output.shape)
    dot = make_dot(output, params = dict(model.named_parameters()))
    outputDirectory.mkdir(parents = True, exist_ok = True)
    path = outputDirectory / f'{modelName} architecture graph'
    outputPath = dot.render(path, format = 'png')
    print(f'File saved to {outputPath}.')
