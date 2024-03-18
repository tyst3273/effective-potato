import h5py 
import numpy as np
import matplotlib.pyplot as plt


f = 'keep.hdf5'

with h5py.File(f,'r') as db:
    etot = db['etotal'][...]
    temp = db['temperature'][...]
    pos = db['positions'][...]
    step = db['steps'][...]

num_atoms = pos.shape[1]

mean_pos = pos.mean(axis=0)
mean_pos -= mean_pos.min()

np.savetxt('mean',mean_pos,fmt='% 6.3f')
print(pos)

stride = 50
temp = temp[::stride]
step = step[::stride]

plt.plot(step,temp)
plt.show()

