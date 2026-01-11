
import numpy as np
import matplotlib.pyplot as plt
import h5py

def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n_lo = db['n_lo'][...]
        n_hi = db['n_hi'][...]
        x_lo = db['x_lo'][...]
        x_hi = db['x_hi'][...]
        v = db['v'][...]
        y = db['y'][...]

    return n_lo, n_hi, x_lo, x_hi, v, y

fig, ax = plt.subplots(1,3,figsize=(8,3),gridspec_kw={'hspace':0.15,'wspace':0.1})

n_lo, n_hi, x_lo, x_hi, v, y = get_data('results_v_sweep_y_0.010.h5')
ax[0].plot(v,n_lo/x_lo,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)
ax[0].plot(v,n_hi/x_hi,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)
ax[0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0].annotate('y=0.01',xy=(0.4,0.85),xycoords='axes fraction',c='k')

n_lo, n_hi, x_lo, x_hi, v, y = get_data('results_v_sweep_y_0.100.h5')
ax[1].plot(v,n_lo/x_lo,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)
ax[1].plot(v,n_hi/x_hi,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)
ax[1].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1].annotate('y=0.01',xy=(0.4,0.85),xycoords='axes fraction',c='k')

n_lo, n_hi, x_lo, x_hi, v, y = get_data('results_v_sweep_y_0.500.h5')
ax[2].plot(v,n_lo/x_lo,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)
ax[2].plot(v,n_hi/x_hi,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)
ax[2].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2].annotate('y=0.01',xy=(0.4,0.85),xycoords='axes fraction',c='k')


ax[0].set_xlabel('v')
ax[1].set_xlabel('v')
ax[2].set_xlabel('v')
ax[0].set_ylabel('\mathcal{I}')

# fig.suptitle(f'y={y:.3f}')

for ii in range(3):

    ax[ii].axis([0,0.5,-0.0175/2,0.175])
    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    
for ii in range(2):
    ax[ii].set_yticklabels([])
  

plt.savefig(f'v_sweep.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

