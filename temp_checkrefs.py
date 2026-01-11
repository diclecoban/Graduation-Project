import h5py
from pathlib import Path
path=Path('data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master/dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat')
with h5py.File(path,'r') as f:
    batch=f['batch']
    cycles_group=f[batch['cycles'][0,0]]
    for idx in [0,5,10,20,40]:
        ref=cycles_group['V'][idx,0]
        d=f[ref]
        print(idx, d.attrs.get('MATLAB_empty', 0), d.shape)
