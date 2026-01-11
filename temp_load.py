import pickle
from pathlib import Path
path=Path('batch1.pkl')
print('loading...')
with path.open('rb') as fp:
    data=pickle.load(fp)
print('cells', len(data))
first_key=next(iter(data))
print('first key', first_key)
print('fields', data[first_key].keys())
