# Semantic Segmentation of Marine Biofouling using Deep Learning

This repository contains a semantic segmentation model for marine biofouling on ship hulls, written in Python with PyTorch. It labels every pixel of an underwater image of a hull as clean hull, soft fouling, hard fouling or background, using an Attention U-Net.

## Project Overview

Biofouling increases the resistance and fuel consumption of ships, and hull inspections are still assessed by eye. The model automates the assessment: from a photograph or a video frame of a hull, it predicts which areas are clean, covered by soft fouling, covered by hard fouling, or background.

Images are annotated in Labelbox and split into training, validation and testing subsets. The network, a U-Net with spatial attention gates on its skip connections and channel attention in every convolution block, is trained on augmented images with a combined cross-entropy and Dice loss, and its learning rate is selected on a grid. The U-Net without attention and a simple convolutional network serve as baselines. Every run is recorded with MLflow, together with its configuration, split, code version and checkpoint. The best model is exported in ONNX format, evaluated on the testing subset, and applied to new images by a Windows application.

Each step is a subcommand of the `biofouling` command line, configured by `configs/default.toml` and the files layered over it. The dataset and the trained models are not part of the repository.

## Repository Structure

- `configs/default.toml`: Every setting, including the classes of the network and the annotation classes each one absorbs.
- `configs/annotationClasses.toml`: The five annotation classes, instead of the four merged classes.
- `configs/similarityAwareSplit.toml`: Split that keeps similar images, such as frames of the same video, in the same subset.
- `biofouling/commandLine.py`: The `biofouling` command line.
- `biofouling/configuration.py`: Layered configuration and its validation.
- `biofouling/data/`: Conversion of the Labelbox export into masks, split of the dataset, data loading and augmentation, and extraction of video frames.
- `biofouling/models/`: Attention U-Net, its attention mechanisms, and the baseline networks.
- `biofouling/training/`: Training loop, loss, learning rate tuning, experiment tracking and export of the model.
- `biofouling/evaluation/`: Metrics, and evaluation of the exported model on the testing subset.
- `biofouling/visualization/`: Masks, training curves, dataset statistics and network summary.
- `application/`: Windows application, which applies an exported model to a folder of images.
- `.env.example`: Template of the Labelbox API key.

## References

1. I. Karlatiras, *Semantic Segmentation of Marine Biofouling Images with Attention U-Net*, Diploma thesis, School of Naval Architecture and Marine Engineering, National Technical University of Athens, 2025. [Link](https://dspace.lib.ntua.gr/xmlui/handle/123456789/62468)
2. O. Bourchas, I. Karlatiras and G. Papalambrou, *A Novel Application of Attention U-Net for Marine Biofouling Segmentation*, Journal of Ocean Engineering and Marine Energy, 2026. [Link](https://link.springer.com/article/10.1007/s40722-026-00535-9)
