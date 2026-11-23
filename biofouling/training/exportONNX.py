import json
import numpy as np
import onnx
import torch
from collections.abc import Sequence
from logging import getLogger, ERROR
from pathlib import Path
from tempfile import TemporaryDirectory
from torch.export import Dim
from torch.nn import Module
from onnxruntime import InferenceSession
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from biofouling.training.checkpoints import loadCheckpoint

logger = getLogger(__name__)
# The exporter warns that the operators of torchvision, which the project does not use, cannot be registered.
getLogger('torch.onnx._internal.exporter._registration').setLevel(ERROR)

def saveONNX(model: Module, resolution: Sequence[int], path: Path, metadata: dict[str, str]) -> Path:
    # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
    # Exporting requires dummy input tensor. Batch size is left dynamic.
    model.eval()
    dummyInput = torch.randn(1, 3, *resolution)
    # The exported graph is optimized, e.g. by constant folding, to improve efficiency.
    # Weights are embedded, so that the model is a single file.
    torch.onnx.export(model, (dummyInput,), path, opset_version = 18, dynamo = True, external_data = False, input_names = ['Input'],
                      output_names = ['Output'], dynamic_shapes = {'x': {0: Dim('Batch')}})

    # The model carries its classes and provenance, which the tester and the Windows application read.
    proto = onnx.load(path)
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, path)

    # Confirm that the exported model reproduces the output of the PyTorch model.
    with torch.no_grad():
        expectedOutput = model(dummyInput).numpy()
    session = InferenceSession(path, providers = ['CPUExecutionProvider'])
    exportedOutput = session.run(None, {'Input': dummyInput.numpy()})[0]
    if not np.allclose(expectedOutput, exportedOutput, rtol = 1e-3, atol = 1e-3):
        raise RuntimeError(f'ONNX model at {path} does not reproduce the PyTorch model.')
    return path

def exportRun(runID: str) -> str:
    # Export the checkpoint of a run, and attach the ONNX model to the same run.
    # Export runs on the CPU, so that the PyTorch reference is computed in full float32 precision, like ONNX Runtime,
    # rather than with the reduced precision (TF32) that GPUs use for convolutions by default.
    client = MlflowClient()
    with TemporaryDirectory() as directory:
        checkpointPath = download_artifacts(run_id = runID, artifact_path = 'checkpoint.pt', dst_path = directory)
        model, checkpoint = loadCheckpoint(checkpointPath, 'cpu')
        metadata = {'classes': json.dumps(checkpoint['classes']), 'architecture': checkpoint['architecture'],
                    'resolution': json.dumps(checkpoint['resolution']), 'splitStrategy': checkpoint['splitStrategy'],
                    'splitHash': checkpoint['splitHash'], 'runID': runID}
        path = Path(directory) / 'model.onnx'
        saveONNX(model, checkpoint['resolution'], path, metadata)
        client.log_artifact(runID, str(path))
    location = f'{client.get_run(runID).info.artifact_uri}/model.onnx'
    logger.info(f'Model of run {runID} exported in ONNX format to {location}.')
    return location
