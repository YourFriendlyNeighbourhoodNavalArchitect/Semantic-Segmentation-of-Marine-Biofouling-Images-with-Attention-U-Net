"""Semantic Segmentation of Marine Biofouling with an Attention U-Net."""
from os import environ

# MLflow prints a hint for coding agents upon import, which only clutters the output. The package is imported before any
# of its modules, so this precedes every import of MLflow.
environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
