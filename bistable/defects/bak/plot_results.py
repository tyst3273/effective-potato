
import numpy as np
import matplotlib.pyplot as plt
import h5py


with h5py.File('results.h5','r') as db:

    n_diff = db['n_diff'][...] # shape = [ny, nv]
    v = db['v'][...]
    y = db['y'][...]

multistable = np.abs(n_diff) > 1e-3

fig, ax = plt.subplots(figsize=(4.5,4.5))

extent = [v.min(),v.max(),y.min(),y.max()]
ax.imshow(multistable,aspect='auto',origin='lower',cmap='Grays',vmin=0,vmax=1,
          interpolation='none',extent=extent)

# ax.axis([0,1,0,0.15])

ax.set_xlabel('v')
ax.set_ylabel('y')

plt.savefig(f'defects_region.png',dpi=200,format='png',bbox_inches='tight')
plt.show()