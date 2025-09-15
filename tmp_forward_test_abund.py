import torch
from src.models.abund_unet2d import Tiny2DUNet

R = 3
model = Tiny2DUNet(R=R, ch=32)
# input shape (batch, R, H, W)
x = torch.randn(1, R, 145, 145)
t = torch.zeros(1,1)
out = model(x, t)
print('output shape:', out.shape)
