
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


blue = '#377eb8' # k
orange = '#ff7f00' # b
green = '#4daf4a' # r

# ------------------------------

def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    v = np.tile(v.reshape(v.size,1),reps=(1,3))
    j = v * n / x 
    j = j.flatten()
    v = v.flatten()

    return n, x, v, j, y, z

# ------------------------------

fig, ax = plt.subplots(1,3,figsize=(8,3),gridspec_kw={'hspace':0.15,'wspace':0.2})

z_list = [0.0]#,0.1,0.5]
colors = [blue,orange,green]

for ii, zz in enumerate(z_list):
    
    c = colors[ii]
    ax[0].plot(-1,0,c=c,lw=2,label=f'z={zz:.3f}')

    n, x, v, j, y, z = get_data(f'results_v_y_0.010_z_{zz:.3f}.h5')
    ax[0].plot(v,j,c=c,lw=0,marker='o',ms=1.5)

    n, x, v, j, y, z = get_data(f'results_v_y_0.090_z_{zz:.3f}.h5')
    ax[1].plot(v,j,c=c,lw=0,marker='o',ms=1.5)

    n, x, v, j, y, z = get_data(f'results_v_y_0.100_z_{zz:.3f}.h5')
    ax[2].plot(v,j,c=c,lw=0,marker='o',ms=1.5)

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.1,0.95))

for ii in range(3):
    _ax = ax[ii]
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize=12)
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.set_rasterization_zorder = 1000000000

y = 0.01
ax[0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k',fontsize=12)
ax[0].annotate(f'y={y:.2f}',xy=(0.05,0.7),xycoords='axes fraction',c='k',fontsize=12)

y = 0.1
ax[1].annotate('(c)',xy=(0.05,0.85),xycoords='axes fraction',c='k',fontsize=12)
ax[1].annotate(f'y={y:.2f}',xy=(0.05,0.7),xycoords='axes fraction',c='k',fontsize=12)

y = 0.25
ax[2].annotate('(e)',xy=(0.05,0.85),xycoords='axes fraction',c='k',fontsize=12)
ax[2].annotate(f'y={y:.2f}',xy=(0.05,0.2),xycoords='axes fraction',c='k',fontsize=12)

ax[0].set_xlabel('v',fontsize=16)
ax[1].set_xlabel('v',fontsize=16)
ax[2].set_xlabel('v',fontsize=16)

for ii in range(3):

    # ax[ii,0].set_yscale('log')
    # ax[ii,1].set_yscale('log')

    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')

    ax[ii].set_ylabel('x',fontsize=16)
    ax[ii].set_ylabel('n',fontsize=16)
    ax[ii].yaxis.set_label_position("right")
    ax[ii].yaxis.tick_right()

    v_lo = 0.0
    v_hi = 0.3
    # ax[ii].axis([v_lo,v_hi,-0.0025,0.04])
    # ax[ii].axis([v_lo,v_hi,-0.02,0.4])

# plt.savefig(f'v_sweep.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

