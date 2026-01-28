
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


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

fig, ax = plt.subplots(1,3,figsize=(8,2.5),gridspec_kw={'hspace':0.15,'wspace':0.3})

z_list = [0.0,0.1,1.0]
colors = ['k','b','r']

for ii, zz in enumerate(z_list):
    
    c = colors[ii]
    ax[0].plot(-1,0,c=c,lw=2,label=f'z={zz:.3f}')

    n, x, v, y, z = get_data(f'results_v_y_0.010_z_{zz:.3f}.h5')

    j = n / x
    j[:,0] *= v
    j[:,1] *= v
    j[:,2] *= v

    ax[0].plot(v,j[:,0],c=c,lw=0,marker='o',ms=0.5)
    # ax[0].plot(v,j[:,1],c=c,lw=0,marker='o',ms=0.5)
    ax[0].plot(v,j[:,2],c=c,lw=0,marker='o',ms=0.5)

    try:
        _lo = np.nanargmax(x[:,1])
    except ValueError:
        _lo = 0
    try:
        _hi = np.nanargmin(x[:,1])
    except ValueError:
        _hi = 0
    ax[0].plot([v[_lo],v[_lo]],[j[_lo,0],j[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[0].plot([v[_hi],v[_hi]],[j[_hi,0],j[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)

    n, x, v, y, z = get_data(f'results_v_y_0.100_z_{zz:.3f}.h5')

    j = n / x
    j[:,0] *= v
    j[:,1] *= v
    j[:,2] *= v

    ax[1].plot(v,j[:,0],c=c,lw=0,marker='o',ms=0.5)
    # ax[1].plot(v,j[:,1],c=c,lw=0,marker='o',ms=0.5)
    ax[1].plot(v,j[:,2],c=c,lw=0,marker='o',ms=0.5)

    try:
        _lo = np.nanargmax(x[:,1])
    except ValueError:
        _lo = 0
    try:
        _hi = np.nanargmin(x[:,1])
    except ValueError:
        _hi = 0
    ax[1].plot([v[_lo],v[_lo]],[j[_lo,0],j[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[1].plot([v[_hi],v[_hi]],[j[_hi,0],j[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)

    n, x, v, y, z = get_data(f'results_v_y_0.250_z_{zz:.3f}.h5')

    j = n / x
    j[:,0] *= v
    j[:,1] *= v
    j[:,2] *= v

    ax[2].plot(v,j[:,0],c=c,lw=0,marker='o',ms=0.5)
    # ax[2].plot(v,j[:,1],c=c,lw=0,marker='o',ms=0.5)
    ax[2].plot(v,j[:,2],c=c,lw=0,marker='o',ms=0.5)

    try:
        _lo = np.nanargmax(x[:,1])
    except ValueError:
        _lo = 0
    try:
        _hi = np.nanargmin(x[:,1])
    except ValueError:
        _hi = 0
    ax[2].plot([v[_lo],v[_lo]],[j[_lo,0],j[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)
    ax[2].plot([v[_hi],v[_hi]],[j[_hi,0],j[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c=c)

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.3,1.05))

y = 0.01
ax[0].annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[0].annotate(f'y={y:.2f}',xy=(0.1,0.75),xycoords='axes fraction',c='k')

y = 0.1
ax[1].annotate('(b)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[1].annotate(f'y={y:.2f}',xy=(0.1,0.75),xycoords='axes fraction',c='k')

y = 0.25
ax[2].annotate('(c)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[2].annotate(f'y={y:.2f}',xy=(0.1,0.75),xycoords='axes fraction',c='k')

ax[0].set_ylabel('j')

for ii in range(3):

    ax[ii].set_xlabel('v')
    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    
    # ax[ii].set_yscale('log')

    v_lo = 0.0
    v_hi = 0.4
    # ax[ii].set_xlim(v_lo,v_hi)
    ax[ii].axis([v_lo,v_hi,-0.005,0.1])
    

plt.savefig(f'iv.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

