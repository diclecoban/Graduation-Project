import pickle
from pathlib import Path
import numpy as np
path=Path('batch1.pkl')
with path.open('rb') as fp:
    data=pickle.load(fp)
cell=data['b1c5']
qd=np.asarray(cell['summary']['QD'])
print('length',len(qd),'min',qd.min(),'max',qd.max())
