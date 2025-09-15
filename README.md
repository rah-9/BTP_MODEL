# UniMix-Diff (tiny CPU demo)

This is a minimal CPU-friendly demo of a semiblind unmixing pipeline combining:
- Tiny STU (unmixing MLP)
- Tiny spectral DDPM (1D)
- Tiny abundance DDPM (2D)
- Simple semiblind posterior sampler (alternating reverse sampling + likelihood gradients)

## Setup
1. Create a venv and install requirements:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

## Run everything (will run on CPU)
python run_all.py

This will:
- generate a small synthetic dataset (data/synthetic)
- train the tiny STU, SpecDM, AbundDM (few epochs for CPU)
- run the semiblind sampler on one sample and print reconstruction MSE

## Notes
- Models and training are intentionally tiny so they are runnable on CPU.
- Once pipeline works, you can increase model sizes and diffusion steps and move to GPU.
# BTP_MODEL
