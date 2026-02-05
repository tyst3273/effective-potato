
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


blue = '#377eb8' # k
orange = '#ff7f00' # b
green = '#4daf4a' # r

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


def get_j_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    return n, x, j, y, z

fig, ax = plt.subplots(1,3,figsize=(8,2.5),gridspec_kw={'hspace':0.15,'wspace':0.15})

z_list = [0.0,0.1,1.0]
colors = [blue,orange,green]

for ii, zz in enumerate(z_list):
    
    c = colors[ii]
    ax[0].plot(-1,0,c=c,lw=2,label=f'z={zz:.3f}')

    n, x, v, y, z = get_data(f'results_v_y_0.010_z_{zz:.3f}.h5')

    j = n / x
    j[:,0] *= v
    j[:,1] *= v
    j[:,2] *= v

    ax[0].plot(v,j[:,0],c=c,lw=0,marker='o',ms=1.5)
    ax[0].plot(v,j[:,1],c=c,lw=1,marker='o',ms=0)
    ax[0].plot(v,j[:,2],c=c,lw=0,marker='o',ms=1.5)

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

    ax[1].plot(v,j[:,0],c=c,lw=0,marker='o',ms=1.5)
    ax[1].plot(v,j[:,1],c=c,lw=1,marker='o',ms=0)
    ax[1].plot(v,j[:,2],c=c,lw=0,marker='o',ms=1.5)

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

    ax[2].plot(v,j[:,0],c=c,lw=0,marker='o',ms=1.5)
    ax[2].plot(v,j[:,1],c=c,lw=1,marker='o',ms=0)
    ax[2].plot(v,j[:,2],c=c,lw=0,marker='o',ms=1.5)

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

    # ----------------------------------------------------------------------------------------------

    # n, x, j, y, z = get_j_data(f'results_j_y_0.010_z_{zz:.3f}.h5')

    # v = x / n
    # v[:,0] *= j
    # v[:,1] *= j
    # v[:,2] *= j

    # ax[0].plot(v[:,0],j,c=c,lw=0,marker='o',ms=0.5)

    # n, x, j, y, z = get_j_data(f'results_j_y_0.100_z_{zz:.3f}.h5')

    # v = x / n
    # v[:,0] *= j
    # v[:,1] *= j
    # v[:,2] *= j

    # ax[1].plot(v[:,0],j,c=c,lw=1,marker='o',ms=0,ls=(0,(2,1)))

    # n, x, j, y, z = get_j_data(f'results_j_y_0.250_z_{zz:.3f}.h5')

    # v = x / n
    # v[:,0] *= j
    # v[:,1] *= j
    # v[:,2] *= j

    # ax[2].plot(v[:,0],j,c=c,lw=0,marker='o',ms=0.5)

    # ----------------------------------------------------------------------------------------------

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.3,1.05))

for ii in range(3):
    _ax = ax[ii]
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize=12)
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.set_rasterization_zorder = 1000000000

ax[1].set_yticklabels([])
ax[2].set_yticklabels([])

y = 0.01
ax[0].annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',c='k',fontsize=12)
ax[0].annotate(f'y={y:.2f}',xy=(0.05,0.8),xycoords='axes fraction',c='k',fontsize=12)

y = 0.1
ax[1].annotate('(b)',xy=(0.05,0.9),xycoords='axes fraction',c='k',fontsize=12)
ax[1].annotate(f'y={y:.2f}',xy=(0.05,0.8),xycoords='axes fraction',c='k',fontsize=12)

y = 0.25
ax[2].annotate('(c)',xy=(0.05,0.9),xycoords='axes fraction',c='k',fontsize=12)
ax[2].annotate(f'y={y:.2f}',xy=(0.05,0.8),xycoords='axes fraction',c='k',fontsize=12)

ax[0].set_ylabel('j',fontsize=16)

for ii in range(3):

    ax[ii].set_xlabel('v',fontsize=16)
    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    
    # ax[ii].set_yscale('log')

    v_lo = 0.0
    v_hi = 0.3
    # ax[ii].set_xlim(v_lo,v_hi)
    ax[ii].axis([v_lo,v_hi,-0.003,0.03])
    

plt.savefig(f'iv.png',dpi=300,format='png',bbox_inches='tight')
plt.show()

