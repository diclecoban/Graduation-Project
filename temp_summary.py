import pickle
from pathlib import Path
path=Path('batch1.pkl')
with path.open('rb') as fp:
    data=pickle.load(fp)
cell=data['b1c0']
summary=cell['summary']
print(summary.keys())
print('QD len', len(summary['QD']))
print('QD first 5', summary['QD'][:5])
