# src/models/abund_unet2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tiny2DUNet(nn.Module):
    """
    Small 2D UNet for abundance maps of shape (R, H, W)
    We model inputs as (batch, R, H, W)
    """
    def __init__(self, R=3, ch=64):
        super().__init__()
        self.inc = nn.Conv2d(R, ch, 3, padding=1)
        self.down = nn.Conv2d(ch, ch*2, 4, stride=2, padding=1)
        # output_padding=1 fixes size mismatch when spatial dims are odd (e.g., 145 -> down -> up)
        self.up = nn.ConvTranspose2d(ch*2, ch, 4, stride=2, padding=1, output_padding=1)
        self.outc = nn.Conv2d(ch, R, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(1, ch), nn.ReLU(), nn.Linear(ch, ch))

    def forward(self, x, t):
        # x: (b, R, H, W)
        b = x.shape[0]
        te = self.time_mlp(t.view(b,1))[:, :, None, None]
        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down(x1))
        x3 = F.relu(self.up(x2) + x1)
        out = self.outc(x3)
        return out
