# Architectures, and the factor by which each one downsamples its input. The input resolution must be a multiple of it,
# so that the decoder restores the input size exactly. Kept apart from the networks, so that the configuration can be
# validated without loading PyTorch.
DOWNSAMPLING_FACTORS = {'Attention U-Net': 16, 'Spatial Attention U-Net': 16, 'U-Net': 16, 'Simple CNN': 8}
