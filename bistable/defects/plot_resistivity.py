
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

        j = db['j'][...]
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

    return n, x, j, y, z

# --------------------------------------------------------------------------------------------------
    
cmap = plt.get_cmap('magma')
norm = np.linspace(0,1,10)
colors = cmap(norm)

fig, ax = plt.subplots(figsize=(4.5,4.5),gridspec_kw={'wspace':0.15,'hspace':0.15})

y = 0.1
z_list = [0,0.1,1.0]
colors = [blue,orange,green]

x_min = 1e9
x_max = 0

for ii, zz in enumerate(z_list):

    ax.plot(-1,0,c=colors[ii],lw=2,label=f'z={zz:.3f}')

    n, x, j, y, z = get_j_data(f'results_j_y_{y:.3f}_z_{zz:.3f}.h5')

    if np.nanmin(x) < x_min:
        x_min = np.nanmin(x)
    if np.nanmax(x) > x_max:
        x_max = np.nanmax(x)
    
    r = x/n
    ax.plot(x[:,0],r[:,0],c=colors[ii],lw=2,marker='o',ms=0)
    ax.plot(x[:,2],r[:,2],c=colors[ii],lw=2,marker='o',ms=0)

# ax.annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',c=blue)
ax.annotate(f'y={y:.2f}',xy=(0.05,0.1),xycoords='axes fraction',c='k')

fig.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(0.1,0.975))

x = np.linspace(x_min,x_max,10001)
r_eq = x / np.exp(-1/x) 
ax.plot(x,r_eq,c='m',ls=(0,(2,1,1,1)),zorder=100)

ax.set_xlabel('x')
ax.set_ylabel('r')

ax.set_yscale('log')
ax.axis([0.075,0.5,2,3000])

plt.savefig(f'resistivity.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

