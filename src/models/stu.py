# src/models/stu.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySTU(nn.Module):
    """
    Tiny unmixing network: per-pixel MLP mapping B-band spectrum -> R abundances.
    Designed for CPU and synthetic experiments.
    """
    def __init__(self, B=64, R=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(B, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, R)
        )
    def forward(self, Y):  # Y: (..., B)
        logits = self.net(Y)
        # abundance constraint: nonneg + sum-to-one via softmax
        A = F.softmax(logits, dim=-1)
        return A
