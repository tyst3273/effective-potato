
import numpy as np
import matplotlib.pyplot as plt
import h5py


with h5py.File('results.h5','r') as db:

    multistable = db['multistable'][...].astype(float) # shape = [ny, nv]
    v = db['v'][...]
    y = db['y'][...]

fig, ax = plt.subplots(figsize=(4.5,4.5))

extent = [v.min(),v.max(),y.min(),y.max()]
ax.imshow(multistable,aspect='auto',origin='lower',cmap='Grays',vmin=0,vmax=1,
          interpolation='none',extent=extent)

ax.set_xlabel('v')
ax.set_ylabel('y')

plt.show()
