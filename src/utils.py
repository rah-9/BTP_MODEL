# src/utils.py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
try:
    from scipy.io import loadmat
    _HAS_SCIPY = True
    try:
        from scipy.optimize import nnls
        _HAS_SCIPY_NNLS = True
    except Exception:
        nnls = None
        _HAS_SCIPY_NNLS = False
except Exception:
    loadmat = None
    _HAS_SCIPY = False
    nnls = None
    _HAS_SCIPY_NNLS = False

def _load_from_mat(mat_path: str):
    """Load expected variables from a .mat file.
    Accepts keys 'hsis', 'abundances', 'endmembers'.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required to load .mat files. Install scipy or provide numpy files")
    m = loadmat(mat_path)
    # Try to find sensible variables. Accept either direct keys or nested struct-like entries.
    hsis = None
    abds = None
    E = None

    # first, look for direct keys
    if 'hsis' in m:
        hsis = m['hsis']
    if 'abundances' in m:
        abds = m['abundances']
    if 'endmembers' in m:
        E = m['endmembers']

    # if hsis not found, try to find any key that looks like an HSI (3D array)
    if hsis is None:
        for k, v in m.items():
            if k.startswith('__'):
                continue
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if arr.ndim == 3:
                # most likely the hyperspectral cube
                hsis = arr
                break

    # if abds or E are missing, we'll attempt simple automatic extraction from HSI
    if hsis is None:
        raise KeyError(f"Could not find a hyperspectral cube inside .mat file {mat_path}. Found keys: {list(m.keys())}")

    hsis = np.asarray(hsis)
    # normalize HSI to [0,1] if values appear to be in a larger integer range
    global LAST_HSI_MAX
    try:
        hsis_max = float(hsis.max())
    except Exception:
        hsis_max = 1.0
    if hsis_max > 1.5:
        # convert to float32 and scale
        hsis = hsis.astype(np.float32) / hsis_max
        LAST_HSI_MAX = hsis_max
    else:
        hsis = hsis.astype(np.float32)
        LAST_HSI_MAX = 1.0
    # Ensure hsis has shape (N, H, W, B). If a single cube (H,W,B) was provided, add batch dim.
    if hsis.ndim == 3:
        hsis = hsis[np.newaxis, ...]
    elif hsis.ndim != 4:
        raise ValueError(f"Loaded hsis has unexpected number of dimensions: {hsis.ndim}. Expected 3 or 4.")

    if E is None:
        # build endmembers via a simple farthest-point sampling on pixel spectra
        def _extract_endmembers_fps(hsi, R=3, seed=0):
            rng = np.random.RandomState(seed)
            # hsi may be (H,W,B) or (N,H,W,B). If batched, use first sample.
            if hsi.ndim == 4:
                cube = hsi[0]
            else:
                cube = hsi
            H, W, B = cube.shape
            pixels = cube.reshape(-1, B)
            # normalize pixels
            pixels = pixels.astype(np.float32)
            # choose first index randomly
            idxs = [rng.randint(0, pixels.shape[0])]
            for _ in range(1, R):
                chosen = pixels[idxs]
                # compute distances to nearest chosen
                dists = np.min(((pixels[:, None, :] - chosen[None, :, :])**2).sum(-1), axis=1)
                idxs.append(int(np.argmax(dists)))
            E = pixels[idxs]
            # normalize spectra
            E = np.clip(E, 0.0, None)
            E = E / (E.max(axis=1, keepdims=True) + 1e-8)
            return E

        # default R guess: 3 (same as demo). If dataset contains gt endmembers in other key, user can provide.
        E = _extract_endmembers_fps(hsis, R=3)

    else:
        E = np.asarray(E)

    if abds is None:
        # approximate abundances via least-squares projection + NN + renorm
        try:
            # approx_fcls expects Y shape (..., B), S shape (R, B) and returns (..., R)
            # hsis currently is (N,H,W,B) so this will return (N,H,W,R)
            A = approx_fcls(hsis, E)
            abds = A
        except Exception as e:
            raise RuntimeError(f"Failed to compute abundance maps from HSI and extracted endmembers: {e}")
    else:
        abds = np.asarray(abds)
        # ensure abds has batch dim (N,H,W,R) if provided as a single H,W,R map
        if abds.ndim == 3:
            abds = abds[np.newaxis, ...]
        elif abds.ndim != 4:
            # allow shape (N, H*W, R) -> reshape if possible
            if abds.ndim == 2:
                # treat as (H*W, R) -> expand batch dim
                HWR = abds.shape[0]
                # can't deduce H,W here; leave as is and let downstream fail with clear shape mismatch
                pass

    return hsis, abds, E

# default scale (set by load_synthetic)
LAST_HSI_MAX = 1.0


def load_synthetic(path='data/synthetic'):
    """
    Load synthetic dataset. `path` may be:
    - a directory containing hsis.npy, abundances.npy, endmembers.npy (original behavior)
    - a single .mat file containing variables 'hsis', 'abundances', 'endmembers'
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() == '.mat':
        return _load_from_mat(str(p))
    if p.is_dir():
        # auto-detect a .mat file inside directory (prefer common names)
        mats = list(p.glob('*.mat'))
        if mats:
            # prefer file named like Indian_pines_corrected.mat if present
            preferred = p / 'Indian_pines_corrected.mat'
            if preferred.exists():
                return _load_from_mat(str(preferred))
            # else pick first .mat
            return _load_from_mat(str(mats[0]))
    # assume directory with npy files
    hsis = np.load(str(p / 'hsis.npy'))
    abds = np.load(str(p / 'abundances.npy'))
    E = np.load(str(p / 'endmembers.npy'))
    return hsis, abds, E

def spec_angle(a,b,eps=1e-8):
    # a,b shape (...,B)
    num = (a*b).sum(-1)
    den = (np.linalg.norm(a,axis=-1)+eps)*(np.linalg.norm(b,axis=-1)+eps)
    cos = np.clip(num/den, -1, 1)
    return np.arccos(cos)

def approx_fcls(Y, S):
    """
    Simple per-pixel nonneg least squares via projecting on nonneg and normalize to sum=1.
    Y: (..., B)
    S: (R, B)
    returns A: (..., R)
    This is approximate: we compute NNLS by solving least squares then ReLU + renorm.
    """
    # reshape
    orig_shape = Y.shape[:-1]
    Yf = Y.reshape(-1, Y.shape[-1])
    R, B = S.shape
    # least squares: A_ls = Y S^T (S S^T)^-1
    St = S
    SS = S @ S.T  # (R,R)
    try:
        inv = np.linalg.inv(SS + 1e-6*np.eye(R))
        A_ls = (Yf @ S.T) @ inv.T
    except Exception:
        A_ls = np.maximum(Yf @ S.T, 0.0)
    A_ls = np.maximum(A_ls, 0.0)
    A_ls = A_ls / (A_ls.sum(axis=-1, keepdims=True)+1e-8)
    A = A_ls.reshape(*orig_shape, R)
    return A


def nnls_fcls(Y, S):
    """
    Solve non-negative least squares per pixel using scipy.optimize.nnls.
    Y: (..., B)
    S: (R, B)
    returns A: (..., R)
    This is slower than `approx_fcls` but yields true NNLS solutions.
    """
    if not _HAS_SCIPY_NNLS:
        raise RuntimeError('scipy.optimize.nnls not available')
    orig_shape = Y.shape[:-1]
    Yf = Y.reshape(-1, Y.shape[-1])
    R, B = S.shape
    A_out = np.zeros((Yf.shape[0], R), dtype=np.float32)
    # nnls expects columns of A as basis, so pass S.T
    St = S.T.copy()
    for i in range(Yf.shape[0]):
        y = Yf[i]
        try:
            x, _ = nnls(St, y)
        except Exception:
            x = np.maximum(np.linalg.lstsq(S.T, y, rcond=None)[0], 0.0)
        # renormalize to sum-to-one if positive
        if x.sum() > 0:
            x = x / (x.sum()+1e-8)
        A_out[i] = x
    return A_out.reshape(*orig_shape, R)
