
import numpy as np
import matplotlib.pyplot as plt
import h5py

def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]
        v = db['v'][...]
        y = db['y'][...]

    n_lo = n[:,0]
    n_hi = n[:,-1]

    x_lo = x[:,0]
    x_hi = x[:,-1]
    
    n_hi[x_hi == 0.0] = np.nan
    x_hi[x_hi == 0.0] = np.nan

    return n_lo, n_hi, x_lo, x_hi, v, y

fig, ax = plt.subplots(3,2,figsize=(4.5,6),gridspec_kw={'hspace':0.15,'wspace':0.1})

n_lo, n_hi, x_lo, x_hi, v, y = get_data('results_v_sweep_y_0.010.h5')

ax[0,0].plot(v,n_lo,c='b',lw=0,marker='o',ms=1)
ax[0,1].plot(v,x_lo,c='r',lw=0,marker='o',ms=1)

ax[0,1].plot(v,x_hi,c='r',lw=0,marker='o',ms=1)
ax[0,0].plot(v,n_hi,c='b',lw=0,marker='o',ms=1)

ax[0,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0,0].annotate('y=0.01',xy=(0.4,0.85),xycoords='axes fraction',c='k')
ax[0,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0,1].annotate('y=0.01',xy=(0.4,0.85),xycoords='axes fraction',c='k')

n_lo, n_hi, x_lo, x_hi, v, y = get_data('results_v_sweep_y_0.100.h5')
ax[1,0].plot(v,n_lo,c='b',lw=0,marker='o',ms=1)
ax[1,0].plot(v,n_hi,c='b',lw=0,marker='o',ms=1)
ax[1,1].plot(v,x_lo,c='r',lw=0,marker='o',ms=1)
ax[1,1].plot(v,x_hi,c='r',lw=0,marker='o',ms=1)
ax[1,0].annotate('(c)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1,0].annotate('y=0.10',xy=(0.4,0.85),xycoords='axes fraction',c='k')
ax[1,1].annotate('(d)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1,1].annotate('y=0.10',xy=(0.4,0.85),xycoords='axes fraction',c='k')

n_lo, n_hi, x_lo, x_hi, v, y = get_data('results_v_sweep_y_0.250.h5')
ax[2,0].plot(v,n_lo,c='b',lw=0,marker='o',ms=1)
ax[2,0].plot(v,n_hi,c='b',lw=0,marker='o',ms=1)
ax[2,1].plot(v,x_lo,c='r',lw=0,marker='o',ms=1)
ax[2,1].plot(v,x_hi,c='r',lw=0,marker='o',ms=1)
ax[2,0].annotate('(e)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2,0].annotate('y=0.50',xy=(0.4,0.85),xycoords='axes fraction',c='k')
ax[2,1].annotate('(f)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2,1].annotate('y=0.50',xy=(0.4,0.85),xycoords='axes fraction',c='k')

ax[2,0].set_xlabel('v')
ax[0,0].set_ylabel('n')

ax[2,1].set_xlabel('v')
ax[0,1].set_ylabel('x')

# fig.suptitle(f'y={y:.3f}')

for ii in range(3):

    ax[ii,0].axis([0,0.5,-0.0175/2,0.175])
    ax[ii,1].axis([0,0.5,-0.06/2,0.6])

    ax[ii,0].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    ax[ii,1].axhline(0,lw=1,ls=(0,(1,1)),c='k')

    ax[ii,1].set_ylabel('x')
    ax[ii,0].set_ylabel('n')
    ax[ii,1].yaxis.set_label_position("right")
    ax[ii,1].yaxis.tick_right()

for ii in range(2):
    ax[ii,0].set_xticklabels([])
    ax[ii,1].set_xticklabels([])

plt.savefig(f'v_sweep.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

