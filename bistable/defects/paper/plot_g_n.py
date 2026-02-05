
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


blue = '#377eb8' # k
orange = '#ff7f00' # b
green = '#4daf4a' # r

# --------------------------------------------------------------------------------------------------

if len(sys.argv) > 1:
    z = float(sys.argv[1])
else:
    z = 0.0

# --------------------------------------------------------------------------------------------------

def get_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    if n.ndim == 1:
        inds = np.argsort(x)
        x = x[inds]
        n = x[inds]
    else:
        for ii in range(n.shape[1]):
            inds = np.argsort(x[:,ii])
            x[:,ii] = x[inds,ii]
            n[:,ii] = n[inds,ii]

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

    # return n, x, v, j, y, z

    return n, x, j, y, z

# --------------------------------------------------------------------------------------------------
    
cmap = plt.get_cmap('magma')
norm = np.linspace(0,1,10)
colors = cmap(norm)

fig, ax = plt.subplots(2,2,figsize=(4.5,4.5),gridspec_kw={'wspace':0.2,'hspace':0.15})

y = 0.1
z_list = [0,0.5,1.0]
colors = [blue,orange,green]

x_min = 1e9
x_max = 0

ax0_ins = ax[0,0].inset_axes([0.05,0.5,0.5,0.45],
                           xlim=(0.0999,0.101),ylim=(4.25e-5,1e-4),yticklabels=[],xticklabels=[])
ax[0,0].indicate_inset_zoom(ax0_ins,edgecolor='black',alpha=1)

# ax1_ins = ax[0,1].inset_axes([0.05,0.4,0.5,0.55],
#                            xlim=(0.0999,0.101),ylim=(4e-4,1e-3),yticklabels=[],xticklabels=[])
# ax[0,1].indicate_inset_zoom(ax1_ins,edgecolor='black',alpha=1)

for ii, zz in enumerate(z_list):

    ax[0,0].plot(-1,0,c=colors[ii],lw=2,label=f'z={zz:.3f}')

    n, x, v, y, z = get_data(f'results_v_y_{y:.3f}_z_{zz:.3f}.h5')

    if np.nanmin(x) < x_min:
        x_min = np.nanmin(x)
    if np.nanmax(x) > x_max:
        x_max = np.nanmax(x)
    
    ax[0,0].plot(x[:,0],n[:,0],c=colors[ii],lw=0,marker='o',ms=1)
    ax[0,0].plot(x[:,2],n[:,2],c=colors[ii],lw=0,marker='o',ms=1)

    ax[0,0].plot(x[:,0],n[:,0],c=colors[ii],lw=0,marker='o',ms=1)
    ax[0,0].plot(x[:,2],n[:,2],c=colors[ii],lw=0,marker='o',ms=1)

    ax0_ins.plot(x[:,0],n[:,0],c=colors[ii],lw=0,marker='o',ms=1)
    ax0_ins.plot(x[:,2],n[:,2],c=colors[ii],lw=0,marker='o',ms=1)

    g = n/x
    ax[0,1].plot(x[:,0],g[:,0],c=colors[ii],lw=0,marker='o',ms=1)
    ax[0,1].plot(x[:,2],g[:,2],c=colors[ii],lw=0,marker='o',ms=1)

    # ax1_ins.plot(x[:,0],g[:,0],c=colors[ii],lw=0,marker='o',ms=1)
    # ax1_ins.plot(x[:,2],g[:,2],c=colors[ii],lw=0,marker='o',ms=1)


for ii, zz in enumerate(z_list):

    n, x, j, y, z = get_j_data(f'results_v_y_{y:.3f}_z_{zz:.3f}.h5')

    ax[1,0].plot(x,n,c=colors[ii],lw=2,marker='o',ms=1)
    ax[1,0].plot(x,n,c=colors[ii],lw=2,marker='o',ms=1)

    g = n/x
    ax[1,1].plot(x,g,c=colors[ii],lw=2,marker='o',ms=1)
    ax[1,1].plot(x,g,c=colors[ii],lw=2,marker='o',ms=1)

ax[0,0].annotate('(a)',xy=(0.6,0.12),xycoords='axes fraction',c='k',fontsize=12)
ax[0,0].annotate(f'y={y:.2f}',xy=(0.6,0.03),xycoords='axes fraction',c='k',fontsize=12)
ax[0,1].annotate('(b)',xy=(0.6,0.12),xycoords='axes fraction',c='k',fontsize=12)
ax[0,1].annotate(f'y={y:.2f}',xy=(0.6,0.03),xycoords='axes fraction',c='k',fontsize=12)
ax[1,0].annotate('(c)',xy=(0.05,0.85),xycoords='axes fraction',c='k',fontsize=12)
ax[1,0].annotate(f'y={y:.2f}',xy=(0.055,0.75),xycoords='axes fraction',c='k',fontsize=12)
ax[1,1].annotate('(d)',xy=(0.05,0.85),xycoords='axes fraction',c='k',fontsize=12)
ax[1,1].annotate(f'y={y:.2f}',xy=(0.055,0.75),xycoords='axes fraction',c='k',fontsize=12)

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.1,0.975))

x = np.linspace(x_min,x_max,10001)
n_eq = np.exp(-1/x)
ax[0,0].plot(x,n_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)
ax[1,0].plot(x,n_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)
ax0_ins.plot(x,n_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)

x = np.linspace(x_min,x_max,10001)
g_eq = np.exp(-1/x) / x
ax[0,1].plot(x,g_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)
ax[1,1].plot(x,g_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)
# ax1_ins.plot(x,g_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)

for ii in range(2):
    for jj in range(2):
        _ax = ax[ii,jj]
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize=12)
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterization_zorder = 1000000000

ax[1,0].set_xlabel('x',fontsize=16)
ax[1,1].set_xlabel('x',fontsize=16)

ax[0,0].set_ylabel('n',fontsize=16)
ax[0,1].set_ylabel('g',fontsize=16)
ax[1,0].set_ylabel('n',fontsize=16)
ax[1,1].set_ylabel('g',fontsize=16)

ax[0,0].set_xticklabels([])
ax[0,1].set_xticklabels([])

ax[0,0].axis([0.1,0.25,0,0.025])
ax[0,1].axis([0.1,0.25,0,0.1])
ax[1,0].axis([0.1,0.25,0,0.025])
ax[1,1].axis([0.1,0.25,0,0.1])

# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')

ax[0,0].axhline(0,lw=1,ls=(0,(1,1)),c='k')
ax[0,1].axhline(0,lw=1,ls=(0,(1,1)),c='k')

ax[0,1].yaxis.set_label_position("right")
ax[0,1].yaxis.tick_right()

ax[1,1].yaxis.set_label_position("right")
ax[1,1].yaxis.tick_right()


plt.savefig(f'n.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

