# src/train_abunddm.py
import torch, numpy as np, os
from src.models.abund_unet2d import Tiny2DUNet
from src.diffusion.ddpm import SimpleDDPM
from src.utils import load_synthetic
from torch import optim

def train_abunddm(epochs=200, lr=2e-4, device='cpu', data_path='data/synthetic'):
    hsis, abds, E = load_synthetic(data_path)
    # Use all abundance maps as training data
    data = torch.from_numpy(abds).permute(0,3,1,2).float().to(device)  # (N,R,H,W)
    model = Tiny2DUNet(R=data.shape[1], ch=32).to(device)
    ddpm = SimpleDDPM(model, T=50, device=device)
    opt = optim.Adam(model.parameters(), lr=lr)
    batch = 8
    for ep in range(epochs):
        perm = torch.randperm(data.shape[0])
        total=0.0
        for i in range(0, data.shape[0], batch):
            xb = data[perm[i:i+batch]]
            t = torch.randint(0, ddpm.T, (xb.shape[0],), device=device)
            noise = torch.randn_like(xb)
            x_t = ddpm.q_sample(xb, t, noise=noise)
            pred = model(x_t, t.float().unsqueeze(1))
            loss = ((pred - noise)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*xb.shape[0]
        if (ep+1)%20==0:
            print(f"[AbundDM] Ep {ep+1}/{epochs} loss {total/data.shape[0]:.6f}")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/abunddm.pth')
    print("Saved AbundDM checkpoint.")

if __name__ == '__main__':
    train_abunddm()
