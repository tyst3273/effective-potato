
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

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    v = np.tile(v.reshape(v.size,1),reps=(1,3))
    v = v.flatten()
    x = x.flatten()
    n = n.flatten()
    j = v * n / x

    inds = np.flatnonzero( ~np.isnan(x) )
    j = j[inds]
    v = v[inds]
    x = x[inds]
    n = n[inds]

    inds = np.argsort(j)
    j = j[inds]
    v = v[inds]
    x = x[inds]
    n = n[inds]

    return n, x, v, j, y, z


fig, ax = plt.subplots(1,3,figsize=(8,2.5),gridspec_kw={'hspace':0.15,'wspace':0.2})

z_list = [0.0,0.5,1.0]
colors = [blue,orange,green]

for ii, zz in enumerate(z_list):
    
    c = colors[ii]
    ax[0].plot(-1,0,c=c,lw=2,label=f'z={zz:.3f}')

    n, x, v, j, y, z = get_j_data(f'results_v_y_0.050_z_{zz:.3f}.h5')
    ax[0].plot(j,v,c=c,lw=2,marker='o',ms=0)

    n, x, v, j, y, z = get_j_data(f'results_v_y_0.100_z_{zz:.3f}.h5')
    ax[1].plot(j,v,c=c,lw=2,marker='o',ms=0)

    n, x, v, j, y, z = get_j_data(f'results_v_y_0.250_z_{zz:.3f}.h5')
    ax[2].plot(j,v,c=c,lw=2,marker='o',ms=0)

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

y = 0.05
ax[0].annotate('(a)',xy=(0.6,0.2),xycoords='axes fraction',c='k',fontsize=12)
ax[0].annotate(f'y={y:.2f}',xy=(0.6,0.1),xycoords='axes fraction',c='k',fontsize=12)

y = 0.1
ax[1].annotate('(b)',xy=(0.6,0.2),xycoords='axes fraction',c='k',fontsize=12)
ax[1].annotate(f'y={y:.2f}',xy=(0.6,0.1),xycoords='axes fraction',c='k',fontsize=12)

y = 0.25
ax[2].annotate('(c)',xy=(0.6,0.2),xycoords='axes fraction',c='k',fontsize=12)
ax[2].annotate(f'y={y:.2f}',xy=(0.6,0.1),xycoords='axes fraction',c='k',fontsize=12)

ax[0].set_ylabel('v',fontsize=16)

ax[1].set_yticklabels([])
ax[2].set_yticklabels([])

for ii in range(3):

    ax[ii].set_xlabel('j',fontsize=16)
    ax[ii].axvline(0,lw=1,ls=(0,(1,1)),c='k',zorder=-1)
    
    # ax[ii].set_yscale('log')

    v_lo = 0.0
    v_hi = 0.5
    # ax[ii].set_xlim(v_lo,v_hi)
    ax[ii].axis([-0.005,0.1,v_lo,v_hi])
    

plt.savefig(f'vi.png',dpi=300,format='png',bbox_inches='tight')
plt.show()

