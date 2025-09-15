# src/train_stu.py
import torch, numpy as np
from torch import nn, optim
from src.models.stu import TinySTU
from src.utils import load_synthetic
import os

def train_stu(epochs=20, lr=1e-3, device='cpu', data_path='data/synthetic'):
    hsis, abds, E = load_synthetic(data_path)
    # flatten pixels: dataset pixels x B
    N, H, W, B = hsis.shape
    R = abds.shape[-1]
    X = hsis.reshape(-1, B).astype('float32')
    A_gt = abds.reshape(-1, R).astype('float32')
    model = TinySTU(B=B, R=R)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X).to(device)
    A_t = torch.from_numpy(A_gt).to(device)
    batch = 1024
    for ep in range(epochs):
        perm = torch.randperm(X_t.shape[0])
        total = 0.0
        for i in range(0, X_t.shape[0], batch):
            idx = perm[i:i+batch]
            xb = X_t[idx]
            ab = A_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, ab)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*xb.shape[0]
        print(f"[STU] Epoch {ep+1}/{epochs} loss {total/X_t.shape[0]:.6f}")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model': model.state_dict(), 'E': E}, 'checkpoints/stu.pth')
    print("Saved STU checkpoint.")
    return model

if __name__ == '__main__':
    train_stu()
