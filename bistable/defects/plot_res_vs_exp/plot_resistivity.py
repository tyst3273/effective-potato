
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from time import sleep
import h5py


blue = '#377eb8'
orange = '#ff7f00'
green = '#4daf4a'

# --------------------------------------------------------------------------------------------------

def get_data(filename):
        
    with h5py.File('../'+filename,'r') as db:

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
        
    with h5py.File('../'+filename,'r') as db:

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

def get_window_avg(data,window_size=50):

    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Convolve the data with the kernel
    data = np.convolve(data, kernel, mode='same')

    return data
# --------------------------------------------------------------------------------------------------

def get_data_prist(filename):

    # TIME[S]       I-SET[A]      V-NVM[V]      V-VMT[A]      TEMP[C]   compliance

    time, current, v4, vT, temp, _ = np.loadtxt(filename,unpack=True,skiprows=1)

    ### intensive ###
    l2 = 0.83 # cm
    l4 = 0.2 # cm
    w = 0.05 # cm
    t = 0.173 # cm
    area = w*t # cm**2
    current /= area # amps / cm^2
    v4 /= l4 # v / cm
    res = v4/current # Ohm * cm
    ### intensive ###

    return time, res, current, temp

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1,2,figsize=(6,3),gridspec_kw={'wspace':0.1})

_time, _res, _current, _temp = get_data_prist('tio2_prist_cond.log')

cut = 11500 
lo_cut = 7500 #3750
_res = get_window_avg(_res)[lo_cut:cut]
_temp = get_window_avg(_temp)[lo_cut:cut]
ax[0].plot(_temp,_res,lw=1.5,c='k',zorder=-10,ls=(0,(4,1,2,1)))

# --------------------

curr, res = np.loadtxt('tio2_curr_vs_res.txt',unpack=True)
coeffs = [441.00745325,  77.24366005]
T = coeffs[0]+curr*coeffs[1]
ax[0].plot(T[:-1],res[:-1],marker='o',ms=3,c=orange,lw=1,ls='--')


ax[0].annotate('equilibrium',xy=(0.5,0.55),xycoords='axes fraction',fontsize=12)
ax[0].annotate('in-situ',xy=(0.4,0.2),xycoords='axes fraction',fontsize=12,c=orange)

# --------------------------------------------------------------------------------------------------

y = 0.1
z_list = [0.1,1.0]
colors = [orange,green]

x_min = 1e9
x_max = 0

for ii, zz in enumerate(z_list):

    ax[1].plot(-1,0,c=colors[ii],lw=2,label=f'z={zz:.3f}')

    n, x, j, y, z = get_j_data(f'results_j_y_{y:.3f}_z_{zz:.3f}.h5')

    if np.nanmin(x) < x_min:
        x_min = np.nanmin(x)
    if np.nanmax(x) > x_max:
        x_max = np.nanmax(x)
    
    r = x/n
    ax[1].plot(x[:,0],r[:,0],c=colors[ii],lw=2,marker='o',ms=0)
    ax[1].plot(x[:,2],r[:,2],c=colors[ii],lw=2,marker='o',ms=0)

x = np.linspace(x_min,x_max,10001)
r_eq = x / np.exp(-1/x) 
ax[1].plot(x,r_eq,c='k',ls=(0,(2,1,1,1)),zorder=100)

ax[1].annotate('equilibrium',xy=(0.55,0.7),xycoords='axes fraction',fontsize=12)
ax[1].annotate('z=0.1',xy=(0.62,0.25),xycoords='axes fraction',fontsize=12,c=orange)
ax[1].annotate('z=1.0',xy=(0.15,0.4),xycoords='axes fraction',fontsize=12,c=green)

# --------------------------------------------------------------------------------------------------

ax[0].annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',fontsize=12)
ax[1].annotate('(b)',xy=(0.85,0.9),xycoords='axes fraction',fontsize=12)

for _ax in ax:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize=12)
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.tick_params(axis='y',which='both',labelcolor='k')
    _ax.set_rasterization_zorder = 1200000000

ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position('right')

# -------------------------

ax[0].set_xlabel('T [C]',fontsize=16)
ax[0].set_ylabel(r'resistivity [$\Omega$-cm]',color='k',fontsize=16)

ax[0].set_ylim(10,1e5)
ax[0].set_xlim(500,950)
ax[0].set_yscale('log')


ax[1].set_ylabel('resistance (non-dim)',fontsize=16,position='right')
ax[1].set_xlabel('x (non-dim)',fontsize=16)

ax[1].set_ylim(5,35)
ax[1].set_xlim(0.15,0.3)
ax[1].set_yscale('log')


# -------------------------

plt.savefig(f'tio2_restivity.png',dpi=300,bbox_inches='tight')
plt.show()