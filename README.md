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

## Run everything (example, CPU)
python run_all.py --input data/synthetic --device cpu

Note: this repository no longer auto-generates synthetic data. Provide a prepared dataset at
`data/synthetic` (or pass `--input <path>`). The dataset should be either:
- a directory containing `hsis.npy`, `abundances.npy`, and `endmembers.npy` (shapes: `(N,H,W,B)`, `(N,H,W,R)`, `(R,B)`),
or
- a single `.mat` file containing variables `hsis`, `abundances`, `endmembers` (or a pack with a 3D HSI and no abds/E where endmembers will be estimated).

Running `run_all.py` will:
- train the tiny STU, SpecDM, AbundDM (few epochs for CPU)
- run the semiblind sampler on the first sample and print reconstruction MSE

## Notes
- Models and training are intentionally tiny so they are runnable on CPU.
- Once pipeline works, you can increase model sizes and diffusion steps and move to GPU.
# BTP_MODEL
