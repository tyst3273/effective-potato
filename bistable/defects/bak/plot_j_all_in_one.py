
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
        
        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    n[x==0.0] = np.nan
    x[x==0.0] = np.nan

    return n, x, j, y, z

fig, ax = plt.subplots(1,2,figsize=(8,3),gridspec_kw={'hspace':0.15,'wspace':0.2})

# y_list = [0.05,0.075,0.1,0.125,0.15] 
# y_list = [0.001,0.005,0.010,0.050,0.100,0.250,0.500]
# y_list = [0.1]
y_list = [0.01,0.1,0.25]

cmap = plt.get_cmap('magma')
norm = np.linspace(0,1,9)
colors = cmap(norm)


for ii, y in enumerate(y_list):

    print(y)

    n, x, j, y, z = get_data(f'results_j_y_{y:.3f}_z_{z:.3f}.h5')
    ax[0].plot(j,n,c=colors[ii],lw=1,ls=(0,(2,1)),marker='o',ms=1)
    ax[1].plot(j,x,c=colors[ii],lw=1,ls=(0,(2,1)),marker='o',ms=1)

    # n, x, j, y, z = get_data(f'results_j_y_{y:.3f}_z_{z:.3f}_newton.h5')
    # ax[0].plot(j,n,c='r',lw=1,ls=(0,(2,1)),marker='o',ms=1)
    # ax[1].plot(j,x,c='r',lw=1,ls=(0,(2,1)),marker='o',ms=1)

# ax[0].annotate('(a)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
# ax[0].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')
# ax[1].annotate('(b)',xy=(0.05,0.85),xycoords='axes fraction',c='k')
# ax[1].annotate(f'y={y:.2f}',xy=(0.4,0.85),xycoords='axes fraction',c='k')

ax[0].set_xlabel('j')
ax[0].set_ylabel('n')

ax[1].set_xlabel('j')
ax[1].set_ylabel('x')

fig.suptitle(f'z={z:.3f}',y=0.925)

# ax[0].set_yscale('log')
# ax[1].set_yscale('log')

# ax[0].axis([0,0.001,-0.004/2,0.04])
# ax[1].axis([0,0.001,-0.04/2,0.4])
# ax[0].axis([0,0.1,-0.004/2,0.04])
# ax[1].axis([0,0.1,-0.04/2,0.4])

ax[0].axhline(0,lw=1,ls=(0,(1,1)),c='k')
ax[1].axhline(0,lw=1,ls=(0,(1,1)),c='k')

ax[1].set_ylabel('x')
ax[0].set_ylabel('n')
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()

# for ii in range(2):
#     ax[ii,0].set_xticklabels([])
#     ax[ii,1].set_xticklabels([])

# fig.legend()

plt.savefig(f'j_sweep_all_in_one.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

