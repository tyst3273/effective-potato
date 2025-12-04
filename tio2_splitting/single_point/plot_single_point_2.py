import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --------------------------------------------------------------------------------------------------

def plot_single_point(hdf5_file='phonon_angmom_calcs.hdf5',num=101):

    fig = plt.figure(figsize=(4.5,6.5))
    gs = plt.GridSpec(2, 2, wspace=0.15,hspace=0.15)
    
    ax1 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,:])

    with h5py.File(hdf5_file, 'r') as db:

        verts = db['verts'][:]
        qpts = db['x_arr'][:]
        freqs = db['freqs'][:]
        freqs = np.sort(freqs, axis=1)
        nq, nb = freqs.shape

        evals = np.zeros((num,nb))
        B = np.zeros(num)
        for ii in range(num):

            _B_vector = db[f'{ii}/B_vector'][:] 
            _evals = np.real(db[f'{ii}/split_freqs'][:])
            _evals = np.sort(_evals)[1,nb:]

            B[ii] = _B_vector[2]
            evals[ii,:] = _evals
        
    for bb in range(nb):
        ax1.plot(B, evals[:,bb], c='k', ms=0.0, marker='o', lw=1)
        # ax3.plot(B, evals[:,bb], c='r', ms=0.0, marker='o', lw=1)
        ax2.plot(B[:-1], np.diff(evals[:,bb])/10, c='k', ms=0.0, marker='o', lw=1)

    ax1.plot(B, evals[:,0], c='r', ms=0.0, marker='o', lw=1)
    ax1.plot(B, evals[:,1], c='r', ms=0.0, marker='o', lw=1)

    ax3.plot(B, evals[:,1]-evals[:,0], c='r', ms=0.0, marker='o', lw=1)

    ax2.plot(B[:-1], np.diff(evals[:,0])/10, c='r', ms=0.0, marker='o', lw=1)
    ax2.plot(B[:-1], np.diff(evals[:,1])/10, c='r', ms=0.0, marker='o', lw=1)

    for _ax in [ax1,ax2,ax3]:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        # _ax.minorticks_on()
        _ax.tick_params(which='both',width=1) #,direction='in')
        _ax.tick_params(which='major',length=5)
        _ax.set_rasterization_zorder = 1000000000

    ax2.set_xlabel(r'B$\parallel$z [T]')

    ax1.set_ylabel('Energy [meV]')
    ax2.set_ylabel(r'|$\partial$E/$\partial$B| [meV/T]')
    ax3.set_ylabel(r'$\Delta$E [meV]')
    
    ax1.annotate(r'TA phonons', xy=(10,9), xycoords='data', c='r')

    ax3.yaxis.set_ticks_position('right')
    ax3.yaxis.set_label_position('right')

    ax1.axis([0,1000,0,50])
    ax2.axis([0,1000,0,0.007])
    ax3.axis([0,1000,0,10])

    fig.suptitle('q=(0,0,1/4)',y=0.93)

    plt.savefig(f'phonon_single_point_2.png',dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------

    plot_single_point(hdf5_file='phonon_angmom_calcs_2.hdf5')

