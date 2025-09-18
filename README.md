# BTP_MODEL: UniMix-Diff - Hyperspectral Unmixing with Diffusion Models

## Overview

BTP_MODEL (B.Tech Project Model) is a comprehensive implementation of hyperspectral unmixing using diffusion models. This project combines spectral and spatial diffusion models with semiblind posterior sampling for advanced hyperspectral image analysis.

## Project Description

This repository implements UniMix-Diff, a semiblind unmixing pipeline that integrates:

- **Tiny STU (Spectral-Temporal Unmixing)**: An MLP-based unmixing network
- **Spectral DDPM**: 1D diffusion model for spectral signatures
- **Abundance DDPM**: 2D diffusion model for spatial abundance maps
- **Semiblind Posterior Sampler**: Alternating reverse sampling with likelihood gradients

The implementation is designed to be CPU-friendly for demonstration and testing purposes, with the capability to scale to GPU for production use.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib (for visualization)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rah-9/BTP_MODEL.git
cd BTP_MODEL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline with CPU:

```bash
python run_all.py --input data/synthetic --device cpu
```

For GPU acceleration (if available):

```bash
python run_all.py --input data/synthetic --device cuda
```

### Custom Dataset

To use your own dataset:

```bash
python run_all.py --input /path/to/your/data --device cpu
```

## Project Structure

```
BTP_MODEL/
├── .vscode/                    # VS Code configuration files
├── __pycache__/               # Python cache files
├── checkpoints/               # Model checkpoints and saved weights
│   ├── stu_model.pth         # Spectral-Temporal Unmixing model weights
│   ├── spec_dm.pth           # Spectral Diffusion Model weights
│   └── abund_dm.pth          # Abundance Diffusion Model weights
├── data/                      # Data directory
│   └── synthetic/            # Synthetic dataset
│       ├── hsis.npy          # Hyperspectral images (N,H,W,B)
│       ├── abundances.npy    # Ground truth abundances (N,H,W,R)
│       └── endmembers.npy    # Endmember spectra (R,B)
├── outputs/                   # Output results and visualizations
│   ├── reconstructions/      # Reconstructed hyperspectral images
│   ├── abundances/           # Estimated abundance maps
│   ├── endmembers/          # Estimated endmember spectra
│   └── evaluation/          # Performance metrics and plots
├── scripts/                   # Utility scripts
│   ├── data_preparation.py   # Data preprocessing utilities
│   ├── evaluation.py         # Evaluation metrics computation
│   └── visualization.py      # Visualization tools
├── src/                       # Source code
│   ├── models/               # Model implementations
│   │   ├── stu.py           # Spectral-Temporal Unmixing network
│   │   ├── diffusion.py     # Diffusion model implementations
│   │   └── samplers.py      # Sampling algorithms
│   ├── training/             # Training utilities
│   │   ├── trainer.py       # Training loops
│   │   └── losses.py        # Loss functions
│   └── utils/                # Utility functions
│       ├── data_loader.py   # Data loading utilities
│       └── metrics.py       # Evaluation metrics
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── run_all.py                # Main execution script
├── tmp_check_shapes.py       # Shape verification utility
├── tmp_debug_abund.py        # Abundance debugging script
├── tmp_forward_test_abund.py # Forward pass testing
├── tmp_import_test.py        # Import testing utility
└── tmp_stats.py              # Statistics computation utility
```

## Data Format

The system supports two data formats:

### NumPy Format (Recommended)
A directory containing three files:
- `hsis.npy`: Hyperspectral images with shape `(N, H, W, B)`
  - N: Number of samples
  - H, W: Spatial dimensions
  - B: Number of spectral bands
- `abundances.npy`: Ground truth abundances with shape `(N, H, W, R)`
  - R: Number of endmembers
- `endmembers.npy`: Endmember spectra with shape `(R, B)`

### MATLAB Format
A single `.mat` file containing variables:
- `hsis`: Hyperspectral images
- `abundances`: Ground truth abundances (optional)
- `endmembers`: Endmember spectra (optional, will be estimated if not provided)

## Training

The training process consists of three main stages:

1. **STU Training**: Trains the spectral-temporal unmixing network
2. **Spectral Diffusion Training**: Trains the 1D spectral diffusion model
3. **Abundance Diffusion Training**: Trains the 2D spatial abundance diffusion model

### Training Parameters

Key parameters can be modified in `run_all.py`:

```python
# Model configuration
STU_HIDDEN_DIM = 128
DIFFUSION_STEPS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 50

# Device configuration
DEVICE = 'cpu'  # or 'cuda' for GPU
```

## Inference

The semiblind posterior sampler performs inference by:

1. Loading pre-trained models
2. Performing initial unmixing with STU
3. Refining results using alternating diffusion sampling
4. Applying likelihood gradients for consistency

### Inference Results

The system outputs:
- Reconstructed hyperspectral images
- Estimated abundance maps
- Estimated endmember spectra
- Reconstruction MSE metrics

## Results

The model performance is evaluated using:

- **Reconstruction MSE**: Mean squared error between original and reconstructed HSI
- **Abundance Error**: Error in estimated abundance maps
- **Spectral Angle Mapper (SAM)**: Spectral similarity metric
- **Root Mean Square Error (RMSE)**: Overall reconstruction quality

Typical results on synthetic data:
- Reconstruction MSE: < 0.01
- Abundance RMSE: < 0.05
- Average SAM: < 5 degrees

## Key Features

### 1. Modular Design
- Separate components for different model types
- Easy to extend and modify individual components
- Clean separation between training and inference

### 2. CPU-Friendly Implementation
- Optimized for CPU execution
- Minimal memory requirements
- Suitable for educational and demonstration purposes

### 3. Scalable Architecture
- Easy migration from CPU to GPU
- Configurable model sizes
- Adjustable diffusion steps for quality vs speed trade-offs

### 4. Comprehensive Evaluation
- Multiple evaluation metrics
- Visualization tools
- Performance monitoring utilities

## Advanced Usage

### Custom Model Configuration

To modify model architecture:

```python
# In run_all.py
model_config = {
    'stu_hidden_dim': 256,
    'diffusion_steps': 200,
    'attention_heads': 8,
    'embedding_dim': 512
}
```

### GPU Acceleration

For GPU usage with larger models:

```bash
# Increase model capacity and use GPU
python run_all.py --input data/synthetic --device cuda --model_size large --diffusion_steps 1000
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or model dimensions
2. **Slow Training**: Consider GPU acceleration or reduce diffusion steps
3. **Poor Results**: Check data format and increase training epochs

### Debug Scripts

Several debugging utilities are provided:

- `tmp_check_shapes.py`: Verify data shapes
- `tmp_debug_abund.py`: Debug abundance estimation
- `tmp_forward_test_abund.py`: Test forward passes
- `tmp_import_test.py`: Test imports
- `tmp_stats.py`: Compute dataset statistics

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Future Work

- [ ] Integration with real hyperspectral datasets
- [ ] Advanced diffusion sampling techniques
- [ ] Multi-scale spatial modeling
- [ ] Real-time inference optimization
- [ ] Web-based demonstration interface

## Citations

If you use this code in your research, please cite:

```bibtex
@misc{btp_model_2024,
  title={BTP_MODEL: UniMix-Diff - Hyperspectral Unmixing with Diffusion Models},
  author={rah-9},
  year={2024},
  url={https://github.com/rah-9/BTP_MODEL}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Research community for hyperspectral imaging advances
- Contributors to diffusion model research

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This is a CPU-optimized demo version. For production use with large datasets, consider GPU acceleration and parameter tuning based on your specific requirements.
