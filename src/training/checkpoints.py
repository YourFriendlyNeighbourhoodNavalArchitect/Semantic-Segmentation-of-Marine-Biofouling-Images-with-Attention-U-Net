from torch import save, load
from src.models.modelFactory import initializeModel

def saveCheckpoint(path, model, architecture, classes, resolution, epoch, splitStrategy, splitHash):
    # PyTorch checkpoint of the best epoch, for later export or fine-tuning.
    # It describes itself (architecture, classes, resolution, split), so that it never depends on the current configuration.
    save({'architecture': architecture, 'classes': classes, 'resolution': list(resolution), 'epoch': epoch,
          'splitStrategy': splitStrategy, 'splitHash': splitHash, 'modelStateDict': model.state_dict()}, path)
    return path

def loadCheckpoint(path, device):
    checkpoint = load(path, map_location = device)
    model = initializeModel(checkpoint['architecture'], numClasses = len(checkpoint['classes']), device = device)
    model.load_state_dict(checkpoint['modelStateDict'])
    return model, checkpoint
