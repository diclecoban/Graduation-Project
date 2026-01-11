import h5py
from pathlib import Path
path = Path('data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master/dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat')
with h5py.File(path,'r') as f:
    Vdlin=f['batch']['Vdlin']
    print(Vdlin.shape, Vdlin.dtype)
