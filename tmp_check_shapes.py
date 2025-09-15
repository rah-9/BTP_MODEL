import sys
sys.path.insert(0, '.')
from src.utils import load_synthetic
hsis, abds, E = load_synthetic('data/synthetic')
print('hsis shape:', hsis.shape)
print('abds shape:', abds.shape)
print('E shape:', E.shape)
