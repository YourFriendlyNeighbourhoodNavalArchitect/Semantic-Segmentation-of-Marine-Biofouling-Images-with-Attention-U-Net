import numpy as np
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from onnx import load, save
from torch import randn, no_grad
from torch.onnx import export
from torch.export import Dim
from onnxruntime import InferenceSession
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from biofouling.training.checkpoints import loadCheckpoint

def saveONNX(model, device, resolution, path, metadata):
    # ONNX offers framework interoperability and shared optimization [https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange].
    # Exporting requires dummy input tensor. Batch size is left dynamic.
    model.eval()
    dummyInput = randn(1, 3, *resolution).to(device)
    # The exported graph is optimized, e.g. by constant folding, to improve efficiency.
    # Weights are embedded, so that the model is a single file.
    export(model, (dummyInput,), path, opset_version = 18, dynamo = True, external_data = False, input_names = ['Input'],
           output_names = ['Output'], dynamic_shapes = {'x': {0: Dim('Batch')}})

    # The model carries its classes and provenance, which the tester and the Windows application read.
    proto = load(path)
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    save(proto, path)

    # Confirm that the exported model reproduces the output of the PyTorch model.
    with no_grad():
        expectedOutput = model(dummyInput).cpu().numpy()
    session = InferenceSession(path, providers = ['CPUExecutionProvider'])
    exportedOutput = session.run(None, {'Input': dummyInput.cpu().numpy()})[0]
    if not np.allclose(expectedOutput, exportedOutput, rtol = 1e-3, atol = 1e-3):
        raise RuntimeError(f'ONNX model at {path} does not reproduce the PyTorch model.')
    return path

def exportRun(runID, device):
    # Export the checkpoint of a run, and attach the ONNX model to the same run.
    client = MlflowClient()
    with TemporaryDirectory() as directory:
        checkpointPath = download_artifacts(run_id = runID, artifact_path = 'checkpoint.pt', dst_path = directory)
        model, checkpoint = loadCheckpoint(checkpointPath, device)
        metadata = {'classes': dumps(checkpoint['classes']), 'architecture': checkpoint['architecture'],
                    'resolution': dumps(checkpoint['resolution']), 'splitStrategy': checkpoint['splitStrategy'],
                    'splitHash': checkpoint['splitHash'], 'runID': runID}
        path = Path(directory) / 'model.onnx'
        saveONNX(model, device, checkpoint['resolution'], path, metadata)
        client.log_artifact(runID, str(path))
    location = f'{client.get_run(runID).info.artifact_uri}/model.onnx'
    print(f'Model of run {runID} exported in ONNX format to {location}.')
    return location
