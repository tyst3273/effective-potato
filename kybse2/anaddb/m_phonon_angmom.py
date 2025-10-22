
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import scipy

ang2bohr = 1.88973
bohr2ang = 1/ang2bohr

ha2eV = 27.2114
ev2ha = 1/ha2eV

ev2meV = 1000
meV2eV = 1/ev2meV

amu2me = 1822.89
me2amu = 1/amu2me

me2kg = 9.10938e-31
meV2thz = 0.2417991
thz2meV = 1/meV2thz

e2C = 1.602176e-19

# --------------------------------------------------------------------------------------------------

def solve_QEP(M,C,K):

    """
    a quadratic eigenvalue problem is defined as Q(E) = E^2 M + E C + K . We want its eigenvectors
    and eigenvalues:
        Q x = 0

    we can solve it by writing an equivalent set of equations
        [  0  I ](  x ) =  E [ I  0 ](  x )
        [ -K -C ]( Ex )      [ 0  M ]( Ex )
    
    and solving by numerical diagonalization.
    """

    rank = M.shape[0]
    I = np.eye(rank)

    A = np.zeros((2*rank,2*rank),dtype=complex)
    A[:rank,rank:] = I
    A[rank:,:rank] = -K
    A[rank:,rank:] = -C

    # ordinary eigenvalue problem
    if np.allclose(M, I):
        evals, evecs = scipy.linalg.eig(A,left=False,right=True)

    # generalized eigenvalue problem
    else:
        ordinary = False
        B = np.zeros((2*rank,2*rank),dtype=complex)
        B[:rank,:rank] = I
        B[rank:,rank:] = M
        evals, evecs = scipy.linalg.eig(A,B,left=False,right=True)

    return evals, evecs
    
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

class c_phonon_angmom:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,phbst_file='run_PHBST.nc',use_nac=False):

        """
        ...
        """

        self.phbst_file = phbst_file
        self.use_nac = use_nac

        self._parse_anaddb_file()
        self._get_qpts_distances()
        self._convert_displacements_to_eigenvectors()

    # ----------------------------------------------------------------------------------------------

    def plot_phonon_angmom(self,scale=1):

        """
        """

        fig, ax = plt.subplots(1,3,figsize=(16,4.5))

        for ii in range(self.num_modes):
            
            ax[0].plot(self.x_arr, self.freqs[:,ii],c='k',lw=1,ms=0)

            _L = self.phonon_angmom[:,ii,0] * scale 
            hi = self.freqs[:,ii] + _L
            lo = self.freqs[:,ii] - _L
            ax[0].fill_between(self.x_arr,lo,hi,color='m',alpha=0.25)

            _L = self.my_phonon_angmom[:,ii,0 ]* scale
            hi = self.freqs[:,ii] + _L
            lo = self.freqs[:,ii] - _L
            ax[0].fill_between(self.x_arr,lo,hi,color='g',alpha=0.25)

        for ii in range(self.num_modes):
            
            ax[1].plot(self.x_arr, self.freqs[:,ii],c='k',lw=1,ms=0)

            _L = self.phonon_angmom[:,ii,1] * scale
            hi = self.freqs[:,ii] + _L
            lo = self.freqs[:,ii] - _L
            ax[1].fill_between(self.x_arr,lo,hi,color='m',alpha=0.25)

            _L = self.my_phonon_angmom[:,ii,1]* scale
            hi = self.freqs[:,ii] + _L
            lo = self.freqs[:,ii] - _L
            ax[1].fill_between(self.x_arr,lo,hi,color='g',alpha=0.25)

        for ii in range(self.num_modes):
            
            ax[2].plot(self.x_arr, self.freqs[:,ii],c='k',lw=1,ms=0)

            _L = self.phonon_angmom[:,ii,2] * scale
            hi = self.freqs[:,ii] + _L
            lo = self.freqs[:,ii] - _L
            ax[2].fill_between(self.x_arr,lo,hi,color='m',alpha=0.25)

            _L = self.my_phonon_angmom[:,ii,2]* scale
            hi = self.freqs[:,ii] + _L
            lo = self.freqs[:,ii] - _L
            ax[2].fill_between(self.x_arr,lo,hi,color='g',alpha=0.25)

        plt.show()
    
    # ----------------------------------------------------------------------------------------------

    def _get_qpts_distances(self):

        """
        ...
        """

        self.qpts_distances = np.zeros(self.num_qpts)
        for ii in range(1,self.num_qpts):
            _dq = np.linalg.norm(self.qpts[ii,:]-self.qpts[ii-1,:])
            self.qpts_distances[ii] = self.qpts_distances[ii-1]+_dq
        
        self.qpts_distances /= self.qpts_distances.max()
        
        self.x_arr = np.linspace(0,1,self.num_qpts)
        
    # ----------------------------------------------------------------------------------------------

    def _parse_anaddb_file(self):

        """
        parse the anaddb PHBST file

        from m_phonons.F90 in abinit:
            !!  Input data is in a.u, whereas the netcdf files saves data in eV for frequencies
            !!  and Angstrom for the displacements
            !!  The angular momentum is output in units of hbar
            
        """

        with nc.Dataset(self.phbst_file,'r') as ds:

            self.types = ds['atom_species'][...]
            self.num_atoms = self.types.size
            self.num_modes = self.num_atoms*3
            self.num_basis = self.num_atoms*3 
            
            self.masses = ds['atomic_mass_units'][...][self.types-1] * amu2me

            self.reduced_pos = ds['reduced_atom_positions'][...] 
            self.lattice_vectors = ds['primitive_vectors'][:,:] # angstrom ?

            # shape = [num_qpts, num_modes, xyz] (its a 3d vector)
            self.phonon_angmom = ds['phangmom'][...] # units of hbar

            self.qpts = ds['qpoints'][...]
            self.num_qpts = self.qpts.shape[0]
            self._get_qpts_distances()

            if self.use_nac:

                pass

            else:

                # shape = [num_qpts, num_basis, num_modes]. units are angstrom
                self.displacements = ds['phdispl_cart'][...,0] + 1j*ds['phdispl_cart'][...,1] 
                self.displacements *= ang2bohr
                self.freqs = ds['phfreqs'][...] * ev2meV

    # ----------------------------------------------------------------------------------------------

    def _convert_displacements_to_eigenvectors(self):
        
        """
        convert displacements to normalized eigenvectors
        """

        _sqrt_m = np.sqrt(self.masses)
        _mass_matrix = np.tile(_sqrt_m.reshape(self.num_atoms,1),reps=(1,3)).flatten()
        _mass_matrix = np.tile(_mass_matrix.reshape(1,self.num_basis),reps=(self.num_basis,1))

        self.eigenvectors = np.zeros( self.displacements.shape,dtype=complex)
        for qq in range( self.num_qpts):

            # convert disp. to eigs: eig_(q nu, a) = sqrt(m_a) disp_(q nu, a)
            self.eigenvectors[qq,...] = _mass_matrix *  self.displacements[qq,...]

            # normalize
            for vv in range( self.num_modes):
                _eig =  self.eigenvectors[qq,:,vv]
                self.eigenvectors[qq,:,vv] /= np.sqrt(_eig.conj() @ _eig) 

        # # check normalization
        # _ovlp = np.zeros((self.num_modes,self.num_modes),dtype=float)
        # for qq in range(self.num_qpts):
        #     _ovlp[...] = 0.0

        #     for uu in range(self.num_modes):
        #         for vv in range(self.num_modes):
        #             _ovlp[uu,vv] = self.eigenvectors[qq,:,uu].conj() @  self.eigenvectors[qq,:,vv]

        #     print(_ovlp.round(6))

    # ----------------------------------------------------------------------------------------------
                
    def average_over_degenerate_modes(self,tol=1e-6):

        """
        average phonon angular momentum over degenerate modes
        """

        self._find_all_degeneracies(tol)

        _ph_angmom = np.zeros(self.phonon_angmom.shape)
        for qq in range(self.num_qpts):

            _num_degen = self.num_degen_manifolds[qq]
            if _num_degen == 0:
                continue

            _manifolds = self.manifolds[qq]
            for _manifold in _manifolds:
                if len(_manifold) > 1:

                    print(self.phonon_angmom[qq,_manifold,...])
                    print(self.phonon_angmom[qq,_manifold,...].mean(axis=0))
                    print('')
                    
                    _ph_angmom[qq,_manifold,...] = \
                        self.phonon_angmom[qq,_manifold,...].mean(axis=0)
                    
        self.phonon_angmom = _ph_angmom

    # ----------------------------------------------------------------------------------------------
                
    def _find_all_degeneracies(self,tol):

        """
        loop over all q-points and find any degeneracies -- only works for unconnected bands since
        this algo assumes that the bands are sorted in ascending order of frequency
        """

        self.num_degen_manifolds = []
        self.manifold_sizes = []
        self.manifold_freqs = []
        self.manifolds = []

        self.has_degeneracies = np.zeros(self.num_qpts,dtype=bool) * False

        for qq in range(self.num_qpts):

            _freqs = self.freqs[qq,:]
            _num_degen, _sizes, _freqs, _manifolds = \
                self._find_degenerate_manifolds(_freqs,tol)
            
            if _num_degen != 0:
                self.has_degeneracies[qq] = True

            self.num_degen_manifolds.append(_num_degen)
            self.manifold_sizes.append(_sizes)
            self.manifold_freqs.append(_freqs)
            self.manifolds.append(_manifolds)

    # ----------------------------------------------------------------------------------------------

    def _find_degenerate_manifolds(self,freqs,tol=1e-6):

        """
        find degenerate modes at the given qpt
        """

        _diff = freqs[1:]-freqs[:-1] 

        manifolds = []
        _manifold = [0]
        for ii in range(self.num_modes-1):

            if _diff[ii] < tol:
                _manifold.append(ii+1)
                if ii == self.num_modes-2:
                    manifolds.append(_manifold)

            else:
                manifolds.append(_manifold)
                _manifold = [ii+1]

        num_manifolds = len(manifolds)
        manifold_sizes = np.zeros(num_manifolds,dtype=int)
        manifold_freqs = np.zeros(num_manifolds,dtype=float)
        for ii in range(num_manifolds):

            manifold_freqs[ii] = freqs[manifolds[ii][0]]

            _n = len(manifolds[ii])
            manifold_sizes[ii] = _n

        num_degen_manifolds = np.count_nonzero(manifold_sizes-1)
        
        # number of denerate manifold, the number of states in each manifold, the frequency of
        # each manifold, and the manifolds
        return num_degen_manifolds, manifold_sizes, manifold_freqs, manifolds
    
    # ----------------------------------------------------------------------------------------------

    def calculate_phonon_angmom(self):

        """
        ...
        """

        
        self.my_phonon_angmom = np.zeros((*self.freqs.shape,3),dtype=float)

        for qq in range(self.num_qpts):
            
            for uu in range(self.num_modes):
                _eu = self.eigenvectors[qq,:,uu].reshape(self.num_atoms,3)

                _L = np.zeros(3,dtype=float)
                for aa in range(self.num_atoms):
                    _L += np.imag(np.cross( _eu[aa,:].conj(), _eu[aa,:]))
                self.my_phonon_angmom[qq,uu,:] = _L
    
    # ----------------------------------------------------------------------------------------------

    def solve_lorentz_dynamical_equations(self,charges=[1,3,-2],B=[0,0,1.0]):

        """
        charges are the ionic charges (oxidation state) of the elements.

        units are m=m_e, Z=e, and w=meV => convert B field to gauss ?
        """

        self.charges = np.array(charges,dtype=float)[self.types-1]
        self.B = np.array(B,dtype=float)

        self.lorentz_matrix = np.zeros(self.eigenvectors.shape,dtype=complex)
        self.dynamical_matrix = np.zeros(self.eigenvectors.shape,dtype=float)

        self.evals = np.zeros((self.num_qpts,self.num_modes*2),dtype=complex)

        for qq in range(self.num_qpts):

            np.fill_diagonal(self.dynamical_matrix[qq,...],self.freqs[qq,:] ** 2)

            for uu in range(self.num_modes):
                _eu = self.eigenvectors[qq,:,uu].conj().reshape(self.num_atoms,3)

                for vv in range(self.num_modes):
                    _ev = self.eigenvectors[qq,:,vv].reshape(self.num_atoms,3)

                    _L = 0.0+0.0j
                    for aa in range(self.num_atoms):
                        _L += self.charges[aa]/self.masses[aa] * np.cross( _eu[aa,:], _ev[aa,:])

                    self.lorentz_matrix[qq,uu,vv] += -1j * ( self.B.dot(_L) )

            _evals, _evecs = solve_QEP(M=np.eye(self.num_modes),C=-self.lorentz_matrix[qq,...],
                            K=-self.dynamical_matrix[qq,...])
                            
            self.evals[qq,:] = _evals

        ### DEV ###
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        for ii in range(self.num_modes*2):

            _w = self.evals[:,ii].real
            flag = (_w <= 0.0)
            c = np.zeros((self.num_qpts,3))
            c[:,0] = 1.0
            c[flag,0] = 0.0
            c[flag,2] = 1.0
                
            ax.scatter(self.x_arr, np.abs(_w) ,lw=0,s=5,marker='o',c=c)

        for ii in range(self.num_modes):
            ax.plot(self.x_arr, self.freqs[:,ii],c='k')

        plt.show()
        ### DEV ###
    
    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    phonon_angmom = c_phonon_angmom()

    # phonon_angmom.average_over_degenerate_modes() # DONT DO THIS
    phonon_angmom.calculate_phonon_angmom()
    phonon_angmom.plot_phonon_angmom()

    # phonon_angmom.solve_lorentz_dynamical_equations(B=[0,0,0.1])
    phonon_angmom.solve_lorentz_dynamical_equations(B=[0,0,100000])











