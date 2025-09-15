# run_all.py
import os
import argparse
from src.data_gen import gen_dataset
from src.train_stu import train_stu
from src.train_specdm import train_specdm
from src.train_abunddm import train_abunddm
from src.semiblind_sampler import semiblind_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input dataset (directory with npy files or .mat file). If not provided, a synthetic dataset will be generated in data/synthetic')
    args = parser.parse_args()

    data_path = args.input or 'data/synthetic'

    # If input not provided, default to data/synthetic. If that directory already contains a .mat file
    # we will use it; otherwise generate synthetic data.
    if args.input is None:
        data_dir = 'data/synthetic'
        os.makedirs(data_dir, exist_ok=True)
        # auto-detection of .mat handled by load_synthetic; check for existence of any .mat file
        has_mat = any([f for f in os.listdir(data_dir) if f.lower().endswith('.mat')])
        if not has_mat:
            print("Generating synthetic data in data/synthetic...")
            gen_dataset(N=80, H=32, W=32, B=64, R=3, noise_std=0.01, seed=0, outdir='data/synthetic')
        else:
            print("Found .mat file in data/synthetic — using it (no generation).")

    print("Training STU (tiny)...")
    train_stu(epochs=30, data_path=data_path)
    print("Training SpecDM (tiny)...")
    train_specdm(epochs=120, data_path=data_path)
    print("Training AbundDM (tiny)...")
    train_abunddm(epochs=120, data_path=data_path)
    print("Running semiblind sampler demo...")
    A_final, S_final, recon = semiblind_run(data_path=data_path)
    print("Done. Visualize results in Python if desired.")


if __name__ == '__main__':
    main()
