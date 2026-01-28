
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    return n, x, j, y, z

fig, ax = plt.subplots(3,2,figsize=(4.5,6),gridspec_kw={'hspace':0.15,'wspace':0.25})

z_list = [0.0,0.1,1.0]
colors = ['k','b','r']

for ii, zz in enumerate(z_list):
    
    c = colors[ii]
    ax[0,0].plot(-1,0,c=c,lw=2,label=f'z={zz:.3f}')

    n, x, j, y, z = get_data(f'results_j_y_0.010_z_{zz:.3f}.h5')

    ax[0,0].plot(j,n[:,0],c=c,lw=0,marker='o',ms=0.5)
    ax[0,1].plot(j,x[:,0],c=c,lw=0,marker='o',ms=0.5)
    # ax[0,0].plot(j,n[:,1],c=c,lw=0,marker='o',ms=0.5)
    # ax[0,1].plot(j,x[:,1],c=c,lw=0,marker='o',ms=0.5)
    # ax[0,0].plot(j,n[:,2],c=c,lw=0,marker='o',ms=0.5)
    # ax[0,1].plot(j,x[:,2],c=c,lw=0,marker='o',ms=0.5)

    try:
        _lo = np.nanargmax(x[:,1])
    except ValueError:
        _lo = 0
    try:
        _hi = np.nanargmin(x[:,1])
    except ValueError:
        _hi = 0
    ax[0,0].plot([j[_lo],j[_lo]],[n[_lo,0],n[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[0,0].plot([j[_hi],j[_hi]],[n[_hi,0],n[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[0,1].plot([j[_lo],j[_lo]],[x[_lo,0],x[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[0,1].plot([j[_hi],j[_hi]],[x[_hi,0],x[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)

    n, x, j, y, z = get_data(f'results_j_y_0.100_z_{zz:.3f}.h5')

    ax[1,0].plot(j,n[:,0],c=c,lw=0,marker='o',ms=0.5)
    ax[1,1].plot(j,x[:,0],c=c,lw=0,marker='o',ms=0.5)
    # ax[1,0].plot(j,n[:,1],c=c,lw=0,marker='o',ms=0.5)
    # ax[1,1].plot(j,x[:,1],c=c,lw=0,marker='o',ms=0.5)
    # ax[1,0].plot(j,n[:,2],c=c,lw=0,marker='o',ms=0.5)
    # ax[1,1].plot(j,x[:,2],c=c,lw=0,marker='o',ms=0.5)

    try:
        _lo = np.nanargmax(x[:,1])
    except ValueError:
        _lo = 0
    try:
        _hi = np.nanargmin(x[:,1])
    except ValueError:
        _hi = 0
    ax[1,0].plot([j[_lo],j[_lo]],[n[_lo,0],n[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[1,0].plot([j[_hi],j[_hi]],[n[_hi,0],n[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[1,1].plot([j[_lo],j[_lo]],[x[_lo,0],x[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[1,1].plot([j[_hi],j[_hi]],[x[_hi,0],x[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)

    n, x, j, y, z = get_data(f'results_j_y_0.250_z_{zz:.3f}.h5')

    ax[2,0].plot(j,n[:,0],c=c,lw=0,marker='o',ms=0.5)
    ax[2,1].plot(j,x[:,0],c=c,lw=0,marker='o',ms=0.5)
    # ax[2,0].plot(j,n[:,1],c=c,lw=0,marker='o',ms=0.5)
    # ax[2,1].plot(j,x[:,1],c=c,lw=0,marker='o',ms=0.5)
    # ax[2,0].plot(j,n[:,2],c=c,lw=0,marker='o',ms=0.5)
    # ax[2,1].plot(j,x[:,2],c=c,lw=0,marker='o',ms=0.5)

    try:
        _lo = np.nanargmax(x[:,1])
    except ValueError:
        _lo = 0
    try:
        _hi = np.nanargmin(x[:,1])
    except ValueError:
        _hi = 0
    ax[2,0].plot([j[_lo],j[_lo]],[n[_lo,0],n[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[2,0].plot([j[_hi],j[_hi]],[n[_hi,0],n[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[2,1].plot([j[_lo],j[_lo]],[x[_lo,0],x[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[2,1].plot([j[_hi],j[_hi]],[x[_hi,0],x[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.1,0.95))

y = 0.01
ax[0,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0,0].annotate(f'y={y:.2f}',xy=(0.6,0.25),xycoords='axes fraction',c='k')
ax[0,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[0,1].annotate(f'y={y:.2f}',xy=(0.6,0.25),xycoords='axes fraction',c='k')

y = 0.1
ax[1,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1,0].annotate(f'y={y:.2f}',xy=(0.6,0.25),xycoords='axes fraction',c='k')
ax[1,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[1,1].annotate(f'y={y:.2f}',xy=(0.6,0.25),xycoords='axes fraction',c='k')

y = 0.25
ax[2,0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2,0].annotate(f'y={y:.2f}',xy=(0.6,0.25),xycoords='axes fraction',c='k')
ax[2,1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
ax[2,1].annotate(f'y={y:.2f}',xy=(0.6,0.25),xycoords='axes fraction',c='k')

ax[2,0].set_xlabel('j')
ax[0,0].set_ylabel('n')

ax[2,1].set_xlabel('j')
ax[0,1].set_ylabel('x')

for ii in range(3):

    # ax[ii,0].set_yscale('log')
    # ax[ii,1].set_yscale('log')

    ax[ii,0].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    ax[ii,1].axhline(0,lw=1,ls=(0,(1,1)),c='k')

    ax[ii,1].set_ylabel('x')
    ax[ii,0].set_ylabel('n')
    ax[ii,1].yaxis.set_label_position("right")
    ax[ii,1].yaxis.tick_right()

    j_lo = 0.0
    j_hi = 0.1
    ax[ii,0].axis([j_lo,j_hi,-0.005,0.1])
    ax[ii,1].axis([j_lo,j_hi,-0.04,0.5])

for ii in range(2):
    ax[ii,0].set_xticklabels([])
    ax[ii,1].set_xticklabels([])

plt.savefig(f'j_sweep.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

