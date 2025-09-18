import argparse
import json
import os
import numpy as np

from src.utils import load_synthetic, LAST_HSI_MAX, approx_fcls, nnls_fcls, spec_angle
from src.semiblind_sampler import semiblind_run


def compute_metrics(Y, A_est, S_est, A_gt, S_gt):
    # Y: (H,W,B) [0,1-scaled], A_est: (H,W,R), S_est: (R,B)
    metrics = {}
    recon = A_est @ S_est
    # rescale using LAST_HSI_MAX (loader sets this global)
    recon_rescaled = recon * LAST_HSI_MAX
    Y_rescaled = Y * LAST_HSI_MAX
    diff = recon_rescaled - Y_rescaled
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    metrics['recon_mse'] = mse
    metrics['recon_rmse'] = rmse

    # abundance RMSE per endmember and global
    diffA = A_est - A_gt
    mse_per_end = np.mean(diffA**2, axis=(0,1))
    rmse_per_end = np.sqrt(mse_per_end)
    metrics['abundance_rmse_per_endmember'] = rmse_per_end.tolist()
    metrics['abundance_rmse_global'] = float(np.sqrt(np.mean(diffA**2)))

    # spectral angle per endmember (radians -> degrees)
    try:
        ang = spec_angle(S_est, S_gt)
        ang_deg = (ang * 180.0 / np.pi)
        metrics['spectral_angle_deg_per_endmember'] = ang_deg.tolist()
        metrics['spectral_angle_deg_mean'] = float(np.mean(ang_deg))
    except Exception:
        metrics['spectral_angle_deg_per_endmember'] = None
        metrics['spectral_angle_deg_mean'] = None

    return metrics


def verdict(metrics, baseline_metrics=None):
    msg = []
    rmse = metrics['recon_rmse']
    sam = metrics.get('spectral_angle_deg_mean', None)
    # simple thresholds (heuristic)
    if rmse < 0.03:
        msg.append(f'Reconstruction RMSE {rmse:.4f} — excellent')
    elif rmse < 0.08:
        msg.append(f'Reconstruction RMSE {rmse:.4f} — acceptable')
    else:
        msg.append(f'Reconstruction RMSE {rmse:.4f} — may need improvement')

    if sam is not None:
        if sam < 5.0:
            msg.append(f'Mean spectral angle {sam:.2f}° — excellent')
        elif sam < 10.0:
            msg.append(f'Mean spectral angle {sam:.2f}° — acceptable')
        else:
            msg.append(f'Mean spectral angle {sam:.2f}° — may need improvement')

    if baseline_metrics is not None:
        b_rmse = baseline_metrics['recon_rmse']
        rel = (b_rmse - rmse) / (b_rmse + 1e-12)
        msg.append(f'Relative RMSE improvement over baseline: {rel*100:.1f}%')

    return '\n'.join(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/synthetic')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sample', type=int, default=0, help='index of sample to evaluate')
    parser.add_argument('--save', action='store_true', help='save metrics to outputs/metrics.json')
    args = parser.parse_args()

    # ensure outputs dir
    os.makedirs('outputs', exist_ok=True)

    hsis, abds, E = load_synthetic(args.input)
    N, H, W, B = hsis.shape
    R = abds.shape[-1]
    idx = args.sample
    if idx < 0 or idx >= N:
        raise ValueError('sample index out of range')

    Y = hsis[idx]
    A_gt = abds[idx]
    S_gt = E

    print('Running semiblind sampler (may use checkpoints)...')
    A_final, S_final, recon = semiblind_run(data_path=args.input, device=args.device)
    # A_final: (H,W,R), S_final: (R,B), recon: (H,W,B)    

    metrics_sampler = compute_metrics(Y, A_final, S_final, A_gt, S_gt)

    # Baseline LS
    try:
        A_ls = nnls_fcls(Y, S_gt)
    except Exception:
        A_ls = approx_fcls(Y, S_gt)
    metrics_ls = compute_metrics(Y, A_ls, S_gt, A_gt, S_gt)

    print('\n=== Sampler metrics ===')
    for k, v in metrics_sampler.items():
        print(f'{k}: {v}')
    print('\n=== LS baseline metrics ===')
    for k, v in metrics_ls.items():
        print(f'{k}: {v}')

    print('\n=== Verdict ===')
    print(verdict(metrics_sampler, baseline_metrics=metrics_ls))

    if args.save:
        out = {'sampler': metrics_sampler, 'ls': metrics_ls}
        with open('outputs/metrics.json', 'w') as f:
            json.dump(out, f, indent=2)
        print('Saved metrics to outputs/metrics.json')


if __name__ == '__main__':
    main()
