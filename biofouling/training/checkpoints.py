import torch
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from torch.nn import Module
from src.models.modelFactory import initializeModel

def saveCheckpoint(path: Path, model: Module, architecture: str, classes: list[dict[str, Any]], resolution: Sequence[int], epoch: int,
                   splitStrategy: str, splitHash: str) -> Path:
    # PyTorch checkpoint of the best epoch, for later export or fine-tuning.
    # It describes itself (architecture, classes, resolution, split), so that it never depends on the current configuration.
    torch.save({'architecture': architecture, 'classes': classes, 'resolution': list(resolution), 'epoch': epoch,
                'splitStrategy': splitStrategy, 'splitHash': splitHash, 'modelStateDict': model.state_dict()}, path)
    return path

def loadCheckpoint(path: Path | str, device: str) -> tuple[Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location = device)
    model = initializeModel(checkpoint['architecture'], numClasses = len(checkpoint['classes']), device = device)
    model.load_state_dict(checkpoint['modelStateDict'])
    return model, checkpoint
