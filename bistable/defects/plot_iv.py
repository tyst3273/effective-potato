
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

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    x[n == 0.0] = np.nan
    n[n == 0.0] = np.nan

    j = np.zeros(x.shape)
    j[:,0] = v * n[:,0] / x[:,0] 
    j[:,1] = v * n[:,1] / x[:,1] 
    j[:,2] = v * n[:,2] / x[:,2] 

    return n, x, v, y, z, j

fig, ax = plt.subplots(1,3,figsize=(8,3),gridspec_kw={'hspace':0.15,'wspace':0.25})

n, x, v, y, z, j = get_data(f'results_v_y_0.010_z_{z:.3f}.h5')
ax[0].plot(v,j,c='m',lw=0,marker='o',ms=1)

try:
    _lo = np.nanargmax(x[:,1])
except ValueError:
    _lo = 0
try:
    _hi = np.nanargmin(x[:,1])
except ValueError:
    _hi = 0
ax[0].plot([v[_lo],v[_lo]],[j[_lo,0],j[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c='m')
# ax[0].plot([v[_hi],v[_hi]],[j[_hi,0],j[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c='m')

ax[0].annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[0].annotate('y=0.01',xy=(0.6,0.7),xycoords='axes fraction',c='k')

n, x, v, y, z, j = get_data(f'results_v_y_0.100_z_{z:.3f}.h5')
ax[1].plot(v,j,c='m',lw=0,marker='o',ms=1)

try:
    _lo = np.nanargmax(x[:,1])
except ValueError:
    _lo = 0
try:
    _hi = np.nanargmin(x[:,1])
except ValueError:
    _hi = 0
ax[1].plot([v[_lo],v[_lo]],[j[_lo,0],j[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c='m')
ax[1].plot([v[_hi],v[_hi]],[j[_hi,0],j[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c='m')

ax[1].annotate('(b)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[1].annotate('y=0.1',xy=(0.6,0.7),xycoords='axes fraction',c='k')

n, x, v, y, z, j = get_data(f'results_v_y_0.250_z_{z:.3f}.h5')
ax[2].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[2].plot(v,j,c='m',lw=0,marker='o',ms=1)
ax[2].annotate('(c)',xy=(0.05,0.9),xycoords='axes fraction',c='k')
ax[2].annotate('y=0.25',xy=(0.6,0.7),xycoords='axes fraction',c='k')

try:
    _lo = np.nanargmax(x[:,1])
except ValueError:
    _lo = 0
try:
    _hi = np.nanargmin(x[:,1])
except ValueError:
    _hi = 0
ax[2].plot([v[_lo],v[_lo]],[j[_lo,0],j[_lo,2]],ms=0,lw=1,ls=(0,(2,1)),c='m')
ax[2].plot([v[_hi],v[_hi]],[j[_hi,0],j[_hi,2]],ms=0,lw=1,ls=(0,(2,1)),c='m')

ax[0].set_xlabel('v')
ax[1].set_xlabel('v')
ax[2].set_xlabel('v')
ax[0].set_ylabel('j')

fig.suptitle(f'z={z:.3f}',y=0.975)

for ii in range(3):
    
    ax[ii].set_yscale('log')
    # ax[ii].axis([0,0.5,1e-43,1])
    # ax[ii].axis([0,0.5,-0.01/2,0.1])
    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    
# for ii in range(1,3):
#     ax[ii].set_yticklabels([])
  

plt.savefig(f'iv.png',dpi=200,format='png',bbox_inches='tight')
plt.show()

