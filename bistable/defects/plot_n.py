
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

if len(sys.argv) > 1:
    z = float(sys.argv[1])
else:
    z = 0.0

def get_v_data(filename):
        
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

        n_lo = db['n_lo'][...]
        x_lo = db['x_lo'][...]
        
        n_hi = db['n_hi'][...]
        x_hi = db['x_hi'][...]

        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    return n_lo, x_lo, n_hi, x_hi, j, y, z
    
cmap = plt.get_cmap('magma')
norm = np.linspace(0,1,10)
colors = cmap(norm)

fig, ax = plt.subplots(figsize=(4.5,4.5))

y_list = [0.001,0.005,0.010,0.050,0.100,0.250,0.500]
z = 0.1

for ii, y in enumerate(y_list):
    n, x, v, y, z = get_v_data(f'results_v_y_{y:.3f}_z_{z:.3f}.h5')
    ax.plot(x,n,c='r',lw=0,marker='o',ms=0.5)

for ii, y in enumerate(y_list):
    n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_y_{y:.3f}_z_{z:.3f}.h5')
    ax.plot(x_lo,n_lo,c='b',lw=0,marker='o',ms=0.5)
    ax.plot(x_hi,n_hi,c='b',lw=0,marker='o',ms=0.5)

# ax.annotate('y=0.01',xy=(0.4,0.9),xycoords='axes fraction',c='k')

ax.set_xlabel('x')
ax.set_ylabel('n')

ax.set_yscale('log')

# ax.axis([0,0.5,-0.01/2,0.1])
ax.axhline(0,lw=1,ls=(0,(1,1)),c='k')


plt.savefig(f'n.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

