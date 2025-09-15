# src/train_specdm.py
import torch, numpy as np, os
from src.models.spec_unet1d import Tiny1DUNet
from src.diffusion.ddpm import SimpleDDPM
from src.utils import load_synthetic
from torch import optim

def train_specdm(epochs=200, lr=2e-4, device='cpu', data_path='data/synthetic'):
    _, _, E = load_synthetic(data_path)
    # E: (R,B) - we'll treat all endmembers (stacked) as dataset
    data = E.astype('float32')
    data = torch.from_numpy(data).to(device)
    # expand to many by small jitter
    data = torch.cat([data, data*0.98 + 0.02*torch.randn_like(data)], dim=0)
    model = Tiny1DUNet(B=data.shape[-1], ch=32)
    model.to(device)
    ddpm = SimpleDDPM(model, T=50, device=device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        idx = torch.randperm(data.shape[0])
        total=0.0
        for i in range(0, data.shape[0], 8):
            batch = data[idx[i:i+8]]
            # sample t
            t = torch.randint(0, ddpm.T, (batch.shape[0],), device=device)
            noise = torch.randn_like(batch)
            x_t = ddpm.q_sample(batch, t, noise=noise)
            pred = model(x_t, t.float().unsqueeze(1))
            loss = ((pred - noise)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*batch.shape[0]
        if (ep+1)%20==0:
            print(f"[SpecDM] Ep {ep+1}/{epochs} loss {total/data.shape[0]:.6f}")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/specdm.pth')
    print("Saved SpecDM checkpoint.")

if __name__ == '__main__':
    train_specdm()
