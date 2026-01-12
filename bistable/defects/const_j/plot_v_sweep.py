
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

    return n, x, v, y, z

fig, ax = plt.subplots(3,2,figsize=(4.5,6),gridspec_kw={'hspace':0.15,'wspace':0.1})

n, x, v, y, z = get_data(f'results_v_sweep_y_0.010_z_{z:.3f}.h5')

ax[0,0].plot(v,n[:,0],c='b',lw=0,marker='o',ms=1)
ax[0,1].plot(v,x[:,0],c='r',lw=0,marker='o',ms=1)
ax[0,0].plot(v,n[:,1],c='b',lw=0,marker='o',ms=1)
ax[0,1].plot(v,x[:,1],c='r',lw=0,marker='o',ms=1)
ax[0,0].plot(v,n[:,2],c='b',lw=0,marker='o',ms=1)
ax[0,1].plot(v,x[:,2],c='r',lw=0,marker='o',ms=1)

ax[0,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0,0].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')
ax[0,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0,1].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')

n, x, v, y, z = get_data(f'results_v_sweep_y_0.100_z_{z:.3f}.h5')

ax[1,0].plot(v,n[:,0],c='b',lw=0,marker='o',ms=1)
ax[1,1].plot(v,x[:,0],c='r',lw=0,marker='o',ms=1)
ax[1,0].plot(v,n[:,1],c='b',lw=0,marker='o',ms=1)
ax[1,1].plot(v,x[:,1],c='r',lw=0,marker='o',ms=1)
ax[1,0].plot(v,n[:,2],c='b',lw=0,marker='o',ms=1)
ax[1,1].plot(v,x[:,2],c='r',lw=0,marker='o',ms=1)

ax[1,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1,0].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')
ax[1,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1,1].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')

n, x, v, y, z = get_data(f'results_v_sweep_y_0.250_z_{z:.3f}.h5')

ax[2,0].plot(v,n[:,0],c='b',lw=0,marker='o',ms=1)
ax[2,1].plot(v,x[:,0],c='r',lw=0,marker='o',ms=1)
ax[2,0].plot(v,n[:,1],c='b',lw=0,marker='o',ms=1)
ax[2,1].plot(v,x[:,1],c='r',lw=0,marker='o',ms=1)
ax[2,0].plot(v,n[:,2],c='b',lw=0,marker='o',ms=1)
ax[2,1].plot(v,x[:,2],c='r',lw=0,marker='o',ms=1)

ax[2,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2,0].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')
ax[2,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2,1].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')

ax[2,0].set_xlabel('v')
ax[0,0].set_ylabel('n')

ax[2,1].set_xlabel('v')
ax[0,1].set_ylabel('x')

fig.suptitle(f'z={z:.3f}',y=0.925)

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

plt.savefig(f'j_sweep.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

