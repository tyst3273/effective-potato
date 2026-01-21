
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

        n_lo = db['n_lo'][...]
        x_lo = db['x_lo'][...]
        
        n_hi = db['n_hi'][...]
        x_hi = db['x_hi'][...]

        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    return n_lo, x_lo, n_hi, x_hi, j, y, z

fig, ax = plt.subplots(1,2,figsize=(8,3),gridspec_kw={'hspace':0.15,'wspace':0.2})

y_list = [0.001,0.005,0.010,0.050,0.100,0.250,0.500]

cmap = plt.get_cmap('magma')
norm = np.linspace(0,1,10)
colors = cmap(norm)

for ii, y in enumerate(y_list):

    print(y)

    n_lo, x_lo, n_hi, x_hi, j, y, z = get_data(f'results_j_y_{y:.3f}_z_{z:.3f}.h5')

    ax[0].plot(j,n_lo,c=colors[ii],lw=1,ls=(0,(2,1)),marker='o',ms=1)
    ax[1].plot(j,x_lo,c=colors[ii],lw=1,ls=(0,(2,1)),marker='o',ms=1)
    ax[0].plot(j,n_hi,c=colors[ii],lw=1,ls=(0,(2,1)),marker='o',ms=1)
    ax[1].plot(j,x_hi,c=colors[ii],lw=1,ls=(0,(2,1)),marker='o',ms=1)

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

plt.savefig(f'j_sweep_all_in_one.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

