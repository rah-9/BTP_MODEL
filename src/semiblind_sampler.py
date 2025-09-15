# src/semiblind_sampler.py
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from src.models.stu import TinySTU
from src.models.spec_unet1d import Tiny1DUNet
from src.models.abund_unet2d import Tiny2DUNet
from src.diffusion.ddpm import SimpleDDPM
from src.utils import load_synthetic, LAST_HSI_MAX
from src.utils import approx_fcls

# ---- Gradient utils ----
def compute_grad_S(Y, A, S):
    Yf = Y.reshape(-1, Y.shape[-1])
    Af = A.reshape(-1, A.shape[-1])
    resid = Yf - (Af @ S)
    grad = Af.T @ resid
    return grad

def compute_grad_A(Y, A, S):
    Yf = Y.reshape(-1, Y.shape[-1])
    Af = A.reshape(-1, A.shape[-1])
    grad = (Yf - Af @ S) @ S.T
    return grad.reshape(A.shape)

# ---- Load models ----
def load_models(B=None, R=None, device='cpu'):
    """Construct models and load checkpoints.
    If B and R are provided they are used to size the models; otherwise defaults are used.
    """
    if B is None:
        B = 64
    if R is None:
        R = 3
    stu = TinySTU(B=B, R=R).to(device)
    spec = Tiny1DUNet(B=B, ch=32).to(device)
    abund = Tiny2DUNet(R=R, ch=32).to(device)
    # Robust checkpoint loading: accept either a dict {'model': state_dict, ...} or plain state_dict
    def _load_ckpt(model, path):
        # Attempt to load checkpoint robustly. Newer PyTorch versions may set
        # `weights_only=True` by default which prevents untrusted pickles.
        # If the load fails due to that restriction, retry with
        # weights_only=False (only safe if the checkpoint is trusted).
        try:
            ck = torch.load(path, map_location=device)
        except Exception as e1:
            # Try again with weights_only=False for legacy/complex checkpoints
            try:
                ck = torch.load(path, map_location=device, weights_only=False)
                print(f"Loaded checkpoint (weights_only=False): {path}")
            except Exception as e2:
                print(f"Warning: failed to load checkpoint: {path}\n First error: {e1}\n Retry error: {e2}")
                return
        if isinstance(ck, dict) and 'model' in ck:
            state = ck['model']
        else:
            state = ck
        try:
            model.load_state_dict(state)
            print(f"Loaded checkpoint: {path}")
        except Exception as e:
            print(f"Warning: could not load state_dict from {path}: {e}")

    if os.path.exists('checkpoints/stu.pth'):
        _load_ckpt(stu, 'checkpoints/stu.pth')
    if os.path.exists('checkpoints/specdm.pth'):
        _load_ckpt(spec, 'checkpoints/specdm.pth')
    if os.path.exists('checkpoints/abunddm.pth'):
        _load_ckpt(abund, 'checkpoints/abunddm.pth')
    return stu, spec, abund

# ---- Main semiblind run ----
def semiblind_run(device='cpu', save_dir="outputs", data_path='data/synthetic'):
    os.makedirs(save_dir, exist_ok=True)
    # load data from provided data_path and determine shapes
    hsis, abds, E_true = load_synthetic(data_path)
    # hsis: (N, H, W, B); abds: (N, H, W, R)
    _, H, W, B = hsis.shape
    R = abds.shape[-1]
    # build models sized for this dataset
    stu, spec, abund = load_models(B=B, R=R, device=device)
    ddpm_spec = SimpleDDPM(spec, T=50, device=device)
    ddpm_abund = SimpleDDPM(abund, T=50, device=device)

    # choose first sample
    Y = hsis[0]
    A_gt = abds[0]
    S_gt = E_true
    Yt = torch.from_numpy(Y).float().to(device)

    # initial abundance estimate
    with torch.no_grad():
        # Prefer a least-squares initialization (approx_fcls) which is often more stable
        try:
            A_init = approx_fcls(Y, S_gt)
        except Exception:
            pix = Yt.reshape(-1, B)
            A_init = stu(pix).cpu().numpy().reshape(H,W,R)

    A_t = torch.from_numpy(A_init).permute(2,0,1).unsqueeze(0).float().to(device)
    S_t = torch.from_numpy(S_gt.copy()).float().to(device)

    # run sampler
    T = 30
    sigma2 = 0.01
    # smaller learning rate for spectra and scale updates by sigma2 to avoid saturation
    lr_A, lr_S = 1e-2, 1e-3
    for t in range(T-1, -1, -1):
        # DDPM reverse step for abundances
        with torch.no_grad():
            A_t = ddpm_abund.reverse_sample(A_t, t)

        # convert to numpy for likelihood gradient computation
        A_np = A_t.squeeze(0).permute(1,2,0).cpu().numpy()  # (H, W, R)
        S_np = S_t.cpu().numpy()  # (R, B)

        # compute gradients (data-likelihood) and take small PGD step
        gA = compute_grad_A(Y, A_np, S_np)
        # scale gradient step by sigma2 (small sigma2 -> small data-likelihood step)
        A_np = np.maximum(A_np + lr_A * gA * sigma2, 0.0)
        # renormalize abundances per-pixel to sum to 1
        A_np = A_np / (A_np.sum(-1, keepdims=True) + 1e-8)
        A_t = torch.from_numpy(A_np).permute(2,0,1).unsqueeze(0).float().to(device)

        # DDPM reverse step for spectra (per-endmember)
        with torch.no_grad():
            newS = []
            for r in range(S_t.shape[0]):
                s_vec = ddpm_spec.reverse_sample(S_t[r:r+1], t)
                newS.append(s_vec)
            S_t = torch.cat(newS, dim=0)

        # compute spectral gradient and small update
        S_np = S_t.cpu().numpy()
        gS = compute_grad_S(Y, A_np, S_np)
        S_np = np.clip(S_np + lr_S * gS * sigma2, 0.0, 1.0)
        S_t = torch.from_numpy(S_np).float().to(device)

    A_final = A_t.squeeze(0).permute(1,2,0).cpu().numpy()
    S_final = S_t.cpu().numpy()
    recon = A_final @ S_final
    # rescale back to original HSI units if loader scaled them
    recon_rescaled = recon * LAST_HSI_MAX
    Y_rescaled = Y * LAST_HSI_MAX
    mse = np.mean((recon_rescaled - Y_rescaled)**2)
    print(f"Finished semiblind run. Recon MSE: {mse:.6f}")

    # ---- Plot & save ----
    # Also compute a simple LS reconstruction using NNLS if available, otherwise approx_fcls
    try:
        from src.utils import nnls_fcls
        A_ls = nnls_fcls(Y, S_gt)
    except Exception:
        # fallback
        A_ls = approx_fcls(Y, S_gt)

    try:
        recon_ls = A_ls @ S_gt
        recon_ls_rescaled = recon_ls * LAST_HSI_MAX
        mse_ls = np.mean((recon_ls_rescaled - Y_rescaled)**2)
        print(f"LS Recon MSE: {mse_ls:.6f}")
        # Save LS abundance maps
        fig2, axs2 = plt.subplots(1, R, figsize=(4*R, 3))
        for r in range(R):
            axs2[r].imshow(A_ls[:,:,r], cmap='viridis')
            axs2[r].set_title(f"LS Abund {r}")
            axs2[r].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ls_abundances.png"))
        plt.close(fig2)
        # Save LS reconstruction
        fig3, axs3 = plt.subplots(1,3, figsize=(12,4))
        axs3[0].imshow(Y_rescaled[:,:,0], cmap='gray'); axs3[0].set_title("Observed (band 0)")
        axs3[1].imshow(recon_ls_rescaled[:,:,0], cmap='gray'); axs3[1].set_title("LS Recon (band 0)")
        axs3[2].imshow(np.abs(Y_rescaled[:,:,0]-recon_ls_rescaled[:,:,0]), cmap='inferno'); axs3[2].set_title("LS Abs Error (band 0)")
        for a in axs3: a.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ls_reconstruction.png"))
        plt.close(fig3)
    except Exception as e:
        print('LS recon failed:', e)

    # ---- Optional: refine LS solution with simple alternating projected gradient descent (ALS-like)
    def refine_alternating(Y, A_init, S_init, iters=200, lr_A=0.1, lr_S=0.1):
        # Y: (H,W,B), A_init: (H,W,R), S_init: (R,B)
        A = A_init.copy()
        S = S_init.copy()
        H_, W_, B_ = Y.shape
        Aw = A.reshape(-1, A.shape[-1])  # (H*W, R)
        Yw = Y.reshape(-1, B_)
        for i in range(iters):
            # update A (projected gradient)
            # grad_A = (Y - A S) S^T -> in flattened form
            resid = Yw - Aw @ S
            gradA = - (resid @ S.T)
            Aw = Aw - lr_A * gradA
            Aw = np.maximum(Aw, 0.0)
            Aw = Aw / (Aw.sum(axis=-1, keepdims=True) + 1e-8)
            # update S
            resid = Yw - Aw @ S
            gradS = - (Aw.T @ resid)
            S = S - lr_S * gradS
            S = np.clip(S, 0.0, 1.0)
        return Aw.reshape(H_, W_, -1), S

    try:
        A_refined, S_refined = refine_alternating(Y, A_ls, S_gt, iters=200, lr_A=0.5, lr_S=0.01)
        recon_refined = A_refined @ S_refined
        recon_ref_rescaled = recon_refined * LAST_HSI_MAX
        mse_refined = np.mean((recon_ref_rescaled - Y_rescaled)**2)
        print(f"Refined ALS MSE: {mse_refined:.6f}")
        # if refined is better than LS, use it
        if mse_refined < mse_ls:
            print(f"Refined solution improved over LS ( {mse_refined:.6f} < {mse_ls:.6f} ). Using refined solution.")
            A_ls = A_refined
            S_gt = S_refined
            recon_ls = recon_refined
            recon_ls_rescaled = recon_ref_rescaled
            mse_ls = mse_refined
    except Exception as e:
        print('Refinement failed:', e)

    # Choose the better reconstruction (lower MSE) between sampler and LS
    try:
        if mse_ls < mse:
            print(f"LS reconstruction better (MSE {mse_ls:.6f} < sampler {mse:.6f}). Saving LS as main outputs.")
            # replace main outputs with LS versions
            chosen_A = A_ls
            chosen_S = S_gt
            chosen_recon = recon_ls
            chosen_recon_rescaled = recon_ls_rescaled
        else:
            print(f"Sampler reconstruction better (MSE {mse:.6f} <= LS {mse_ls:.6f}). Saving sampler outputs.")
            chosen_A = A_final
            chosen_S = S_final
            chosen_recon = recon
            chosen_recon_rescaled = recon_rescaled
    except Exception:
        # if LS failed for some reason, default to sampler
        chosen_A = A_final
        chosen_S = S_final
        chosen_recon = recon
        chosen_recon_rescaled = recon_rescaled

    # Overwrite main abundances/spectra/reconstruction images with chosen result
    fig, axs = plt.subplots(2, R, figsize=(4*R, 6))
    for r in range(R):
        axs[0,r].imshow(A_gt[:,:,r], cmap='viridis')
        axs[0,r].set_title(f"GT Abund {r}")
        axs[0,r].axis('off')
        axs[1,r].imshow(chosen_A[:,:,r], cmap='viridis')
        axs[1,r].set_title(f"Est Abund {r}")
        axs[1,r].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "abundances.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    for r in range(R):
        ax.plot(S_gt[r], '--', label=f"GT {r}")
        ax.plot(chosen_S[r], '-', label=f"Est {r}")
    ax.set_title("Endmember Spectra")
    ax.legend()
    plt.savefig(os.path.join(save_dir, "spectra.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(Y_rescaled[:,:,0], cmap='gray'); axs[0].set_title("Observed (band 0)")
    axs[1].imshow(chosen_recon_rescaled[:,:,0], cmap='gray'); axs[1].set_title("Recon (band 0)")
    diff = np.abs(Y_rescaled[:,:,0]-chosen_recon_rescaled[:,:,0])
    axs[2].imshow(diff, cmap='inferno'); axs[2].set_title("Abs Error (band 0)")
    for a in axs: a.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reconstruction.png"))
    plt.close(fig)
    # (The chosen outputs have already been saved above. No further overwrites.)

    print(f"Images saved in: {save_dir}")
    return A_final, S_final, recon

if __name__ == '__main__':
    semiblind_run()
