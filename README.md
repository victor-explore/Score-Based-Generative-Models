Here's a markdown README for your GitHub repository that you can directly copy and paste:

```markdown
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
```

## VQ-VAE Configuration

```python
dimension_of_codebook_vectors = 128
number_of_codebook_vectors = 1024
commitment_cost = 1
```

## Usage

### Training NCSN
```python
model = Unet(model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# Train the model with custom noise schedule
```

### Training VQ-VAE Enhanced NCSN
```python
vqvae = VQVAE(embedding_dim=embedding_dim, num_embeddings=num_embeddings)
latent_unet = Unet(model_config).to(device)
# Train on VQ-VAE latent representations
```

### Sampling
```python
# Standard NCSN sampling
images = ALD(model, sigmas=sigmas, num_samples=100)

# VQ-VAE enhanced sampling
images = ALD_vqvae(model, sigmas=sigmas, num_samples=100)
```

## Features
- Configurable architecture components
- Multiple noise schedules
- FID score evaluation
- Support for different datasets
- Modular implementation for easy modifications

## Requirements
- PyTorch
- torchvision
- numpy
- scipy
- tqdm
- matplotlib

## References
- Adapted from https://github.com/explainingai-code/DDPM-Pytorch
- Implementation based on Score-Based Generative Modeling research

## License
[MIT License](LICENSE)
```
