
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def set_grids(ax,major=1,minor=0.05):

    x_major_locator = MultipleLocator(major)
    y_major_locator = MultipleLocator(major)

    x_minor_locator = MultipleLocator(minor)
    y_minor_locator = MultipleLocator(minor)

    x_major_formatter = FormatStrFormatter('%2.1f')
    y_major_formatter = FormatStrFormatter('%2.1f')

    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.xaxis.set_major_formatter(x_major_formatter)

    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_minor_locator(y_minor_locator)
    ax.yaxis.set_major_formatter(y_major_formatter)

    ax.xaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
    ax.xaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
    ax.yaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
    ax.yaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
    ax.xaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
    ax.xaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
    ax.yaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
    ax.yaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
    ax.xaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
    ax.xaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)
    ax.yaxis.grid(color='g',ls='-',lw=1,which='major',alpha=0.75)
    ax.yaxis.grid(color='g',ls='--',lw=0.5,which='minor',alpha=0.75)

def configure_ax(ax,grids=False):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    if grids:
        set_grids(ax)

    ax.minorticks_on()
    ax.tick_params(which='both',width=1,labelsize='large')
    ax.tick_params(which='major',length=5)
    ax.tick_params(which='minor',length=2)

# --------------------------------------------------------------------------------------------------


cmap = 'magma'
interp = 'nearest'
vmin = 0
vmax = 2

full = 'reduced_data/rutile_293K_quenched.hdf5'


# --------------------------------------------------------------------------------------------------
# do full dataset

with h5py.File(full,'r') as db:
    full_sig = db['signal'][...]
    full_H = db['H'][...]
    full_K = db['K'][...]
    full_L = db['L'][...]


extent = [full_H.min(),full_H.max(),full_L.min(),full_L.max()]
K = [0.0,0.16]
for K in K:
    ind = np.flatnonzero(full_K >= K).min()

    fig, ax = plt.subplots(figsize=(8,6),gridspec_kw={'wspace':0.15})

    ax.imshow(full_sig[:,ind,:].T,aspect='auto',origin='lower',vmin=vmin,vmax=vmax,
                cmap=cmap,interpolation=interp,extent=extent)

    configure_ax(ax)
    
    ax.set_xlabel('H (rlu)',fontsize='x-large')
    ax.set_ylabel('L (rlu)',fontsize='x-large')

    ax.annotate(f'K={K:3.2f} (rlu)',xy=(-5,2.5),xycoords='data',fontsize='x-large',
        annotation_clip=False,color=(0.2,0.9,0.2),fontweight='bold')

    ax.axis([-5,5,-3,3])

    plt.savefig(f'full_K{K:3.2f}.png',dpi=150,bbox_inches='tight')
    plt.clf()
    plt.close()


# --------------------------------------------------------------------------------------------------
# do summed dataset

summed = 'reduced_data/293K_quenched_summed.hdf5'

cmap = 'magma'
interp = 'nearest'
vmin = 0.2
vmax = 0.75


with h5py.File(summed,'r') as db:
    summed_sig = db['signal'][...]
    summed_H = db['h'][...]
    summed_K = db['k'][...]
    summed_L = db['l'][...]


extent = [summed_H.min(),summed_H.max(),summed_L.min(),summed_L.max()]
for ind, K in enumerate(summed_K):

    fig, ax = plt.subplots(figsize=(6,6),gridspec_kw={'wspace':0.15})

    ax.imshow(summed_sig[:,ind,:].T,aspect='auto',origin='lower',vmin=vmin,vmax=vmax,
                cmap=cmap,interpolation=interp,extent=extent)

    configure_ax(ax)

    ax.set_xlabel('h (rlu)',fontsize='x-large')
    ax.set_ylabel('l (rlu)',fontsize='x-large')

    ax.annotate(f'k={K:3.2f} (rlu)',xy=(-0.475,0.45),xycoords='data',fontsize='x-large',
        annotation_clip=False,color=(0.2,0.9,0.2),fontweight='bold')

    ax.axis([-0.5,0.5,-0.5,0.5])

    plt.savefig(f'summed_K{K:3.2f}.png',dpi=150,bbox_inches='tight')
    plt.clf()
    plt.close()
    















