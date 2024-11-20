# Score-Based Generative Models Implementation

This repository contains implementations of Score-Based Generative Models, including a Noise Conditioned Score Network (NCSN) and a VQ-VAE enhanced NCSN model. The models are implemented in PyTorch and trained on image datasets.

## Models

### 1. Noise Conditioned Score Network (NCSN)
- UNet-based architecture for score estimation
- Annealed Langevin dynamics sampling
- FID score evaluation using Inception v3
- Support for different noise level schedules

### 2. VQ-VAE Enhanced NCSN
- Vector Quantized Variational Autoencoder (VQ-VAE) for learning compressed image representations
- NCSN trained on VQ-VAE's latent space
- Modified sampling procedure for generating images through the VQ-VAE decoder

## Architecture Details

### NCSN Architecture
- Down blocks with residual connections
- Mid blocks with self-attention
- Up blocks with skip connections
- Configurable channels, layers, and attention heads

### VQ-VAE Architecture
- Encoder: ConvNet with batch normalization
- Vector Quantizer with commitment loss
- Decoder: Transposed convolutions for image reconstruction

## Training Configuration

```python
model_config = {
    'im_channels': 3,
    'im_size': 128,
    'down_channels': [32, 64, 128, 256, 256],
    'mid_channels': [256, 256, 256],
    'down_sample': [True, True, True, False],
    'time_emb_dim': 256,
    'num_down_layers': 1,
    'num_mid_layers': 1,
    'num_up_layers': 1,
    'num_heads': 16
}
