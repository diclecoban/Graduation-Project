import pickle
path='batch1.pkl'
with open(path,'rb') as fp:
    data=pickle.load(fp)
cycle=data['b1c0']['cycles']['0']
print('dQdV first entries', cycle['dQdV'][:5])
