# src/data_gen.py
import numpy as np
import os

def gen_endmembers(R=3, B=64, seed=0):
    rng = np.random.RandomState(seed)
    # create smooth spectral curves by mixing Gaussians
    x = np.linspace(0, 1, B)
    E = []
    for r in range(R):
        centers = rng.uniform(0,1, size=(2,))
        amps = rng.uniform(0.5,1.0, size=(2,))
        s = sum(a * np.exp(-((x-c)**2)/(2*(0.05+r*0.01)**2)) for a,c in zip(amps,centers))
        s = s + 0.02 * rng.randn(B)
        s = np.clip(s, 0.0, None)
        s = s / (s.max()+1e-8)
        E.append(s)
    E = np.stack(E, axis=0)  # (R,B)
    return E

def gen_dataset(N=80, H=32, W=32, B=64, R=3, noise_std=0.01, seed=0, outdir='data/synthetic'):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    E = gen_endmembers(R=R, B=B, seed=seed)
    hsis = []
    abds = []
    for i in range(N):
        # generate abundance maps per pixel via Dirichlet
        alphas = rng.uniform(0.5,3.0, size=(R,))
        A = rng.dirichlet(alphas, size=(H*W,))  # (H*W, R)
        A = A.reshape(H, W, R)
        # mix
        Y = A @ E  # (H,W,B) via broadcasting (A: H W R) dot (R B)
        # Add noise
        Y = Y + noise_std * rng.randn(H, W, B)
        Y = np.clip(Y, 0.0, 1.0)
        hsis.append(Y.astype(np.float32))
        abds.append(A.astype(np.float32))
    hsis = np.stack(hsis, axis=0)
    abds = np.stack(abds, axis=0)
    np.save(os.path.join(outdir, 'hsis.npy'), hsis)
    np.save(os.path.join(outdir, 'abundances.npy'), abds)
    np.save(os.path.join(outdir, 'endmembers.npy'), E.astype(np.float32))
    print(f"Saved dataset: {outdir} | hsis {hsis.shape} abds {abds.shape} endmembers {E.shape}")

if __name__ == '__main__':
    gen_dataset()
