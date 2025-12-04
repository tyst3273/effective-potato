import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --------------------------------------------------------------------------------------------------

def plot_splitting(hdf5_file='phonon_angmom_calcs.hdf5',index=0, 
                   bounds=[[3, 4, 58, 61],[3, 4, 4, 10]]):

    fig = plt.figure(figsize=(6.5,6.5))
    gs = plt.GridSpec(2, 2,wspace=0.225,hspace=0.15)
    
    ax1 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])

    with h5py.File(hdf5_file, 'r') as db:

        verts = db['verts'][:]
        qpts = db['x_arr'][:]
        freqs = db['freqs'][:]
        freqs = np.sort(freqs, axis=1)
        nq, nb = freqs.shape

        B_vector = db[f'{index}/B_vector'][:] 

        evals = np.real(db[f'{index}/split_freqs'][:])
        evals = np.sort(evals, axis=1)[:,nb:]
        
        # labels = db['labels'][:]
        
    for bb in range(nb):

        ax1.plot(qpts, freqs[:,bb], c='k', ms=0.0, marker='o', lw=1)
        ax2.plot(qpts, freqs[:,bb], c='k', ms=0.15, marker='o', lw=0.5)
        ax3.plot(qpts, freqs[:,bb], c='k', ms=0.15, marker='o', lw=0.5)
        
        ax1.plot(qpts, evals[:,bb], c='r', ms=0.0, marker='o', lw=1)
        ax2.plot(qpts, evals[:,bb], c='r', ms=0.25, marker='o', lw=0.5)
        ax3.plot(qpts, evals[:,bb], c='r', ms=0.25, marker='o', lw=0.5)

    for vv in verts:
        ax1.axvline(x=vv, c='k', ls='--', lw=0.75)
        ax2.axvline(x=vv, c='k', ls='--', lw=0.75)
        ax3.axvline(x=vv, c='k', ls='--', lw=0.75)

    for _ax in [ax1, ax2, ax3]:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        # _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize=10) #,direction='in')
        _ax.tick_params(which='major',length=5)
        # _ax.tick_params(which='minor',length=2)
        _ax.set_rasterization_zorder = 1000000000

    labels = [r'$\Gamma$','X','M',r'$\Gamma$','Z','R','A','Z','R','X','M','A']

    ax1.axis([0,1,0,110])
    ax1.set_xticks(verts)
    ax1.set_xticklabels(labels)
    
    xlo, xhi, ylo, yhi = bounds[0]
    ax2.set_xticks([verts[xlo],verts[xhi]])
    ax2.set_xticklabels([labels[xlo],labels[xhi]])
    ax2.axis([verts[xlo],verts[xhi],ylo,yhi])
    rect = patches.Rectangle((verts[xlo], ylo), verts[xhi]-verts[xlo], yhi-ylo, 
                             linewidth=2, edgecolor='b', facecolor='none', zorder=10)
    ax1.add_patch(rect)
    ax1.annotate(f'(b)', fontweight='bold',xy=(verts[xlo]-0.1,yhi+1), xycoords='data', c='b')

    xlo, xhi, ylo, yhi = bounds[1]
    ax3.set_xticks([verts[xlo],verts[xhi]])
    ax3.set_xticklabels([labels[xlo],labels[xhi]])
    ax3.axis([verts[xlo],verts[xhi],ylo,yhi])
    rect = patches.Rectangle((verts[xlo], ylo), verts[xhi]-verts[xlo], yhi-ylo, 
                             linewidth=2, edgecolor='b', facecolor='none', zorder=10)
    ax1.add_patch(rect)

    ax1.set_ylabel('Energy [meV]')
    fig.suptitle(f'B=({B_vector[0]:.0f},{B_vector[1]:.0f},{B_vector[2]:.0f}) [T]',y=0.935)
    ax1.annotate(f'(c)', fontweight='bold',xy=(verts[xlo]-0.1,yhi+1), xycoords='data', c='b')

    ax1.annotate(f'(a)',xy=(-0.1,0.975),xycoords='axes fraction', c='k')
    ax2.annotate(f'(b)',xy=(1.015,0.95),xycoords='axes fraction', c='k')
    ax3.annotate(f'(c)',xy=(1.015,0.95),xycoords='axes fraction', c='k')

    plt.savefig(f'phonon_splitting_B_{B_vector[0]:.0f}_{B_vector[1]:.0f}_{B_vector[2]:.0f}_T.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------

    index = 0
    bounds = [[3, 4, 58, 61],
              [3, 4, 4, 10]]
    plot_splitting(hdf5_file='phonon_angmom_all_calcs.hdf5', index=index, bounds=bounds)

    index = 1
    bounds = [[3, 4, 58, 61],
              [3, 4, 4, 10]]
    plot_splitting(hdf5_file='phonon_angmom_all_calcs.hdf5', index=index, bounds=bounds)

    index = 2
    bounds = [[3, 4, 58, 61],
              [3, 4, 4, 10]]
    plot_splitting(hdf5_file='phonon_angmom_all_calcs.hdf5', index=index, bounds=bounds)

    # ----------------------------------------------------------------------------------------------

    index = 3
    bounds = [[4, 6, 55, 62],
              [7, 9, 55, 62]]
    plot_splitting(hdf5_file='phonon_angmom_all_calcs.hdf5', index=index, bounds=bounds)

    index = 4
    bounds = [[4, 6, 55, 62],
              [7, 9, 55, 62]]
    plot_splitting(hdf5_file='phonon_angmom_all_calcs.hdf5', index=index, bounds=bounds)

    index = 5
    bounds = [[4, 6, 55, 62],
              [7, 9, 55, 62]]
    plot_splitting(hdf5_file='phonon_angmom_all_calcs.hdf5', index=index, bounds=bounds)

    # ----------------------------------------------------------------------------------------------

    # index = 0
    # bounds = [[4, 6, 55, 62],
    #           [7, 9, 55, 62]]
    # plot_splitting(hdf5_file='phonon_angmom_Y_calcs.hdf5', index=index, bounds=bounds)

    # index = 1
    # bounds = [[4, 6, 55, 62],
    #           [7, 9, 55, 62]]
    # plot_splitting(hdf5_file='phonon_angmom_Y_calcs.hdf5', index=index, bounds=bounds)

    # index = 2
    # bounds = [[4, 6, 55, 62],
    #           [7, 9, 55, 62]]
    # plot_splitting(hdf5_file='phonon_angmom_Y_calcs.hdf5', index=index, bounds=bounds)

    # ----------------------------------------------------------------------------------------------

