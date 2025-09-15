import sys
sys.path.insert(0,'.')
from src.utils import load_synthetic
import numpy as np
hsis, abds, E = load_synthetic('data/synthetic')
print('hsis dtype, min,max:', hsis.dtype, hsis.min(), hsis.max())
print('abds dtype, min,max:', abds.dtype, abds.min(), abds.max())
print('E dtype, min,max:', E.dtype, E.min(), E.max())
print('hsis mean std:', hsis.mean(), hsis.std())
print('E mean std:', E.mean(), E.std())
print('abds sum per pixel min/max:', abds.sum(-1).min(), abds.sum(-1).max())
