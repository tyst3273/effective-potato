
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

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

# --------------------------------------------------------------------------------------------------
    
cmap = plt.get_cmap('magma')
norm = np.linspace(0,1,10)
colors = cmap(norm)

fig, ax = plt.subplots(1,1,figsize=(4.5,4.5),gridspec_kw={'wspace':0.15,'hspace':0.15})

y = 0.1
z_list = [0,0.1,1.0]
colors = ['k','b','r']

x_min = 1e9
x_max = 0

for ii, zz in enumerate(z_list):

    ax.plot(-1,0,c=colors[ii],lw=2,label=f'z={zz:.3f}')

    n, x, v, y, z = get_data(f'results_v_y_{y:.3f}_z_{zz:.3f}.h5')

    if np.nanmin(x) < x_min:
        x_min = np.nanmin(x)
    if np.nanmax(x) > x_max:
        x_max = np.nanmax(x)
 
    g = n/x
    ax.plot(x[:,0],g[:,0],c=colors[ii],marker='o',ms=0.0,lw=2)

# ax.annotate('(a)',xy=(0.65,0.15),xycoords='axes fraction',c='k')
ax.annotate(f'y={y:.2f}',xy=(0.8,0.9),xycoords='axes fraction',c='k')

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.1,0.975))

x = np.linspace(x_min,x_max,10001)
g_eq = np.exp(-1/x) / x
ax.plot(x,g_eq,c='m',ls=(0,(2,1,1,1)),zorder=100,lw=2)

ax.set_xlabel('x')
ax.set_ylabel('g')

# ax.set_yscale('log')
ax.axis([0.09999,0.1005,4.525e-4,5.25e-4])

# ax.axis([0.1,0.3,0,0.05])
# ax.axhline(0,lw=1,ls=(0,(1,1)),c='k')

plt.savefig(f'g_zoom.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

