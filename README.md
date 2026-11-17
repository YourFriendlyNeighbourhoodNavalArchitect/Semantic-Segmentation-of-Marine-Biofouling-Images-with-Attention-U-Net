# Semantic Segmentation of Marine Biofouling using Deep Learning

This repository contains a semantic segmentation model for marine biofouling on ship hulls, written in Python with PyTorch. It labels every pixel of an underwater image with a fouling class, using an Attention U-Net.

## Project Overview

Biofouling increases the resistance and fuel consumption of ships, and hull inspections are still assessed by eye. The model automates the assessment: from a photograph or a video frame of a hull, it predicts which areas are clean, covered by soft fouling, covered by hard fouling, or background.

Annotated images are split into training, validation and testing subsets. The network, a U-Net with attention mechanisms that focus on the relevant regions and features of the image, is trained on augmented images, with its learning rate selected on a grid. The best model is exported in ONNX format, evaluated on the testing subset, and applied to new images by a Windows application.

The dataset and the trained models are not part of the repository.

## Repository Structure

- `configs/`: Settings. `default.toml` holds every setting; further files override part of it.
- `src/biofouling/data/`: Conversion of the annotations, split of the dataset, data loading and augmentation, and extraction of video frames.
- `src/biofouling/models/`: Attention U-Net, its building blocks, and baseline networks.
- `src/biofouling/training/`: Training loop, loss, metrics, learning rate tuning, experiment tracking and export of the model.
- `src/biofouling/evaluation/`: Evaluation of the exported model.
- `src/biofouling/visualization/`: Masks, training curves, dataset statistics and network graph.
- `Application/`: Windows application, which applies an exported model to a folder of images.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The exact version of every package is pinned in `uv.lock`.

```bash
uv sync
```

On Linux, PyTorch includes CUDA 13, which requires an NVIDIA driver of version 580 or newer and a GPU of compute capability 7.5 or higher (Turing or newer). On Windows and macOS, PyTorch runs on CPU, which suffices for evaluation and export. The network summary (`biofouling summary`) requires the optional dependencies (`uv sync --extra visualization`) and Graphviz.

## Data

Data are stored in `data/`, and outputs in `outputs/`:

```text
data/
  images/          Annotated images, named <ID>.jpg
  labels.ndjson    Labelbox export
  masks/           Masks of the annotation classes, named <ID>.npy (biofouling labels)
  metadata.json    Classes and similar images of every image (biofouling labels)
  splits/          Image IDs of each subset, one file per split strategy (biofouling split)
  videos/          Videos, and the frames extracted from them (biofouling frames)
outputs/
  mlflow.db        Runs, parameters and metrics
  mlruns/          Files attached to the runs: checkpoints, exported models, plots
  optuna.db        Learning rate tuning studies
  predictions/     Predictions of the testing subset, per run
  visualizations/  Dataset statistics and network graph
```

The Labelbox API key is read from `.env` (see `.env.example`).

## Usage

Every command reads `configs/default.toml`. Further configuration files are layered over it with `--config`, e.g. to distinguish the five annotation classes instead of the merged classes:

```bash
uv run biofouling train --config configs/annotationClasses.toml
```

1. Convert the Labelbox export into masks and metadata:

   ```bash
   uv run biofouling labels
   ```

2. Split the dataset. Images are split one by one, stratified with respect to their classes. With `--config configs/groupedSplit.toml`, similar images are kept in the same subset instead; pass the same file to the commands that follow.

   ```bash
   uv run biofouling split
   ```

3. Train with the configured learning rate, or evaluate the grid of learning rates and keep the best. Either command exports the model in ONNX format and prints its location. An interrupted grid resumes where it stopped.

   ```bash
   uv run biofouling train
   ```

   ```bash
   uv run biofouling tune
   ```

4. Evaluate the exported model of a run on its testing subset. The results are attached to the run.

   ```bash
   uv run biofouling test --run <run ID>
   ```

5. Compare runs in the MLflow interface:

   ```bash
   uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
   ```

Further commands: `biofouling frames` (extract video frames), `biofouling view` (browse augmented images and masks), `biofouling graphs` (dataset statistics), `biofouling export` (export the checkpoint of a run) and `biofouling summary` (network summary and graph). `uv run biofouling --help` lists them all.

## References

1. I. Karlatiras, *Semantic Segmentation of Marine Biofouling Images with Attention U-Net*, Diploma thesis, School of Naval Architecture and Marine Engineering, National Technical University of Athens, 2025. [Link](https://dspace.lib.ntua.gr/xmlui/handle/123456789/62468)
2. O. Bourchas, I. Karlatiras and G. Papalambrou, *A Novel Application of Attention U-Net for Marine Biofouling Segmentation*, Journal of Ocean Engineering and Marine Energy, 2026. [Link](https://link.springer.com/article/10.1007/s40722-026-00535-9)
