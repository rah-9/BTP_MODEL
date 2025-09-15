import sys
sys.path.insert(0, '.')
try:
    import src.utils as utils
    import src.semiblind_sampler as sampler
    print('Imports OK')
except Exception as e:
    print('Import failed:', e)
    raise
