import pickle
from pathlib import Path
path=Path('batch1.pkl')
with path.open('rb') as fp:
    data=pickle.load(fp)
cell=data['b1c0']
cycles=cell['cycles']
cycle0=cycles['0']
print(cycle0.keys())
print(type(cycle0['V']), cycle0['V'][:5])
