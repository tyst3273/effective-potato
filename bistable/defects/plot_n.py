
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
    

fig, ax = plt.subplots(figsize=(4.5,4.5))

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.100_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.150_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.200_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.250_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.300_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.350_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)

# n_lo, x_lo, n_hi, x_hi, j, y, z = get_j_data(f'results_j_sweep_y_0.400_z_{z:.3f}.h5')
# ax.plot(x_lo,n_lo,c='r',lw=0,marker='o',ms=1)


n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.100_z_{z:.3f}.h5')
ax.plot(x,n,c='r',lw=0,marker='o',ms=0.5)

n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.150_z_{z:.3f}.h5')
ax.plot(x,n,c='g',lw=0,marker='o',ms=0.5)

n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.200_z_{z:.3f}.h5')
ax.plot(x,n,c='b',lw=0,marker='o',ms=0.5)

n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.250_z_{z:.3f}.h5')
ax.plot(x,n,c='m',lw=0,marker='o',ms=0.5)

n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.300_z_{z:.3f}.h5')
ax.plot(x,n,c='k',lw=0,marker='o',ms=0.5)

n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.350_z_{z:.3f}.h5')
ax.plot(x,n,c='c',lw=0,marker='o',ms=0.5)

n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.400_z_{z:.3f}.h5')
ax.plot(x,n,c='orange',lw=0,marker='o',ms=0.5)

# n, x, v, y, z = get_v_data(f'results_v_sweep_y_0.010_z_{z:.3f}.h5')
# ax.plot(x,n,c='b',lw=0,marker='o',ms=0.5)

# ax.plot(x_hi,n_hi,c='k',lw=0,marker='o',ms=0.5)


# ax.annotate('y=0.01',xy=(0.4,0.9),xycoords='axes fraction',c='k')

ax.set_xlabel('x')
ax.set_ylabel('n')

# ax.set_yscale('log')

ax.axis([0,0.5,-0.01/2,0.1])
ax.axhline(0,lw=1,ls=(0,(1,1)),c='k')


plt.savefig(f'n.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

