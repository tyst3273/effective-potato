
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

if len(sys.argv) > 1:
    z = float(sys.argv[1])
else:
    z = 0.0

def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    j = np.zeros(x.shape)
    j[:,0] = v * n[:,0] / x[:,0] 
    j[:,1] = v * n[:,1] / x[:,1] 
    j[:,2] = v * n[:,2] / x[:,2] 

    return n, x, v, y, z, j

fig, ax = plt.subplots(1,3,figsize=(8,3),gridspec_kw={'hspace':0.15,'wspace':0.1})

n, x, v, y, z, j = get_data(f'results_v_sweep_y_0.010_z_{z:.3f}.h5')
ax[0].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[0].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[0].annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[0].annotate('y=0.01',xy=(0.4,0.9),xycoords='axes fraction',c='k')

n, x, v, y, z, j = get_data(f'results_v_sweep_y_0.100_z_{z:.3f}.h5')
ax[1].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[1].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[1].annotate('(b)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[1].annotate('y=0.1',xy=(0.4,0.9),xycoords='axes fraction',c='k')

n, x, v, y, z, j = get_data(f'results_v_sweep_y_0.250_z_{z:.3f}.h5')
ax[2].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[2].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[2].annotate('(c)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[2].annotate('y=0.25',xy=(0.4,0.9),xycoords='axes fraction',c='k')


ax[0].set_xlabel('v')
ax[1].set_xlabel('v')
ax[2].set_xlabel('v')
ax[0].set_ylabel('j')

# fig.suptitle(f'y={y:.3f}')

for ii in range(3):

    ax[ii].axis([0,0.5,-0.015/2,0.15])
    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    
for ii in range(1,3):
    ax[ii].set_yticklabels([])
  

plt.savefig(f'iv.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

