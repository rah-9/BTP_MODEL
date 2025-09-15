# src/diffusion/ddpm.py
import torch
import torch.nn.functional as F
import numpy as np

class SimpleDDPM:
    def __init__(self, model, T=50, device='cpu'):
        """
        model: nn.Module that predicts noise given x_t and timestep scalar t (tensor shape [B, ...])
        """
        self.model = model
        self.T = T
        self.device = device
        betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cum = np.cumprod(alphas)
        self.betas = torch.tensor(betas, device=device)
        self.alphas = torch.tensor(alphas, device=device)
        self.alphas_cum = torch.tensor(alphas_cum, device=device)

    def q_sample(self, x0, t, noise=None):
        # x0: tensor, t: int tensor scalar or vector length batch
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cum = self.alphas_cum[t].sqrt().view(-1, *([1]*(x0.dim()-1)))
        sqrt_one_minus = (1 - self.alphas_cum[t]).sqrt().view(-1, *([1]*(x0.dim()-1)))
        return sqrt_alphas_cum * x0 + sqrt_one_minus * noise

    def p_mean_variance(self, x_t, t):
        # predict noise
        eps_pred = self.model(x_t, t.float())
        # compute mean for posterior step (simplified DDPM single-step Euler)
        alpha_t = self.alphas_cum[t].view(-1, *([1]*(x_t.dim()-1)))
        alpha_prev = self.alphas_cum[t-1].view(-1, *([1]*(x_t.dim()-1))) if (t>0).any() else torch.ones_like(alpha_t)
        beta_t = self.betas[t].view(-1, *([1]*(x_t.dim()-1)))
        mean = (1/alpha_t.sqrt())*(x_t - beta_t/(1 - alpha_t).sqrt()*eps_pred)
        return mean, eps_pred

    def sample_noise(self, shape):
        return torch.randn(shape, device=self.device)

    def reverse_sample(self, x_t, t_idx):
        """
        Single simplified reverse step (not exact DDPM numerics but works for demo).
        t_idx: integer scalar (0..T-1), we move to t-1
        """
        t = torch.tensor([t_idx], device=self.device)
        mean, eps = self.p_mean_variance(x_t, t)
        if t_idx > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        return mean + noise * 0.0  # no extra noise for stability in tiny demo

