import h5py
from pathlib import Path
import numpy as np
path=Path('data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master/dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat')
with h5py.File(path,'r') as f:
    batch=f['batch']
    cycles_group=f[batch['cycles'][0,0]]
    idx=5
    V=np.array(f[cycles_group['V'][idx,0]])[0]
    Qd=np.array(f[cycles_group['Qd'][idx,0]])[0]
    print(V[:5])
    print(Qd[:5])
