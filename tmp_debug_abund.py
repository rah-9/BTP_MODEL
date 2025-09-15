import sys
sys.path.insert(0,'.')
from src.semiblind_sampler import semiblind_run
from src.utils import load_synthetic, approx_fcls
hsis, abds, E = load_synthetic('data/synthetic')
Y = hsis[0]
A_ls = approx_fcls(Y, E)
print('A_ls stats (min,max,mean):')
for r in range(A_ls.shape[-1]):
    a=A_ls[:,:,r]
    print(r, a.min(), a.max(), a.mean())
# run sampler (this will print its own MSEs and save images)
A_final, S_final, recon = semiblind_run(data_path='data/synthetic')
print('A_final stats:')
for r in range(A_final.shape[-1]):
    a=A_final[:,:,r]
    print(r, a.min(), a.max(), a.mean())
print('S_final stats: min,max,mean per endmember')
for r in range(S_final.shape[0]):
    s=S_final[r]
    print(r, s.min(), s.max(), s.mean())
