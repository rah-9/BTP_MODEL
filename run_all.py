# run_all.py
import os
import argparse
from src.train_stu import train_stu
from src.train_specdm import train_specdm
from src.train_abunddm import train_abunddm
from src.semiblind_sampler import semiblind_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input dataset (directory with npy files or .mat file). If not provided, a synthetic dataset will be generated in data/synthetic')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    data_path = args.input or 'data/synthetic'
    # Note: this repository no longer auto-generates synthetic data. Provide a prepared
    # dataset at `data_path` (either a directory with hsis.npy / abundances.npy / endmembers.npy
    # or a .mat file with expected variables). See README for details.

    print("Training STU (tiny)...")
    train_stu(epochs=30, data_path=data_path, device=args.device)
    print("Training SpecDM (tiny)...")
    train_specdm(epochs=120, data_path=data_path, device=args.device)
    print("Training AbundDM (tiny)...")
    train_abunddm(epochs=120, data_path=data_path, device=args.device)
    print("Running semiblind sampler demo...")
    A_final, S_final, recon = semiblind_run(data_path=data_path, device=args.device)
    print("Done. Visualize results in Python if desired.")


if __name__ == '__main__':
    main()
