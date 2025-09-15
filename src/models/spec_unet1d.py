# src/models/spec_unet1d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiny1DUNet(nn.Module):
    """
    Very small 1D UNet for spectral vectors (B,)
    Input shape: (batch, B)
    We'll treat B as channels length-1 and use conv1d with fake spatial dim.
    """
    def __init__(self, B=64, ch=64):
        super().__init__()
        self.enc1 = nn.Conv1d(1, ch, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(ch, ch*2, kernel_size=3, padding=1)
        self.dec1 = nn.Conv1d(ch*2, ch, kernel_size=3, padding=1)
        self.out = nn.Conv1d(ch, 1, kernel_size=3, padding=1)
        # time embedding simple linear (we'll concat later)
        self.time_mlp = nn.Sequential(nn.Linear(1, ch), nn.ReLU(), nn.Linear(ch, ch))

    def forward(self, x, t):
        # x: (B,) or (batch, B)
        if x.dim()==1:
            x = x.unsqueeze(0)
        b = x.shape[0]
        x = x.unsqueeze(1)  # (b,1,B)
        te = self.time_mlp(t.view(b,1))[:, :, None]  # (b,ch,1)
        e1 = F.relu(self.enc1(x))  # (b,ch,B)
        e2 = F.relu(self.enc2(e1))  # (b,2ch,B)
        d1 = F.relu(self.dec1(e2) + e1)
        out = self.out(d1)  # (b,1,B)
        out = out.squeeze(1)
        return out
