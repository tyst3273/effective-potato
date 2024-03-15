import h5py 
import numpy as np
import matplotlib.pyplot as plt


f = 'nvt.hdf5'

with h5py.File(f,'r') as db:
    etot = db['etotal'][...]
    temp = db['temperature'][...]
    pos = db['positions'][...]

num_atoms = pos.shape[1]

mean_pos = pos.mean(axis=0)
mean_pos -= mean_pos.min()

np.savetxt('mean',mean_pos,fmt='% 6.3f')
print(pos)

