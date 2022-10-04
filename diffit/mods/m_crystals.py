

import numpy as np


class c_rutile:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,basis=None,types=None,lattice_vectors=None,charges=None):

        """
        define the primitive unitcell;

        WARNING: nothing is really error checked here
        """

        # types of atoms
        if types is None:
            self.types = np.array([0,0,1,1,1,1])
        else:
            self.types = np.array(types)

        self.num_basis = self.types.size
    
        # positions of atoms in reduced coordinates
        if basis is None:
            self.basis = np.array([[0.5000000000000000,  0.5000000000000000,  0.5000000000000000],
                                   [0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
                                   [0.1953400114833092,  0.8046599885166907,  0.5000000000000000],
                                   [0.8046599885166907,  0.1953400114833092,  0.5000000000000000],
                                   [0.3046599885166907,  0.3046599885166907,  0.0000000000000000],
                                   [0.6953400114833093,  0.6953400114833093,  0.0000000000000000]])
        else:
            self.basis = np.array(basis)

        # lattice vectors for crystal translations
        if lattice_vectors is None:
            self.lattice_vectors = np.array([[4.593,0.000,0.000],
                                             [0.000,4.593,0.000],
                                             [0.000,0.000,2.959]])
        else:
            self.lattice_vectors = np.array(lattice_vectors)

        # charges for MD potential
        if charges is None:
            self.charges = np.array([2.196,-1.098])
        else:
            self.charges = np.array(charges,dtype=float)

    # ----------------------------------------------------------------------------------------------

    def make_supercell(self,reps=[1,1,1]):

        """
        create rectangular supercell by replicating primitive cell reps[ii] number of times 
        along the ii^th direction 
        """

        self.reps = np.array(reps)
        self.num_reps = np.prod(self.reps)

        self.sc_lattice_vectors = np.copy(self.lattice_vectors)
        self.sc_lattice_vectors[0,:] *= self.reps[0]
        self.sc_lattice_vectors[1,:] *= self.reps[1]
        self.sc_lattice_vectors[2,:] *= self.reps[2]

        self.num_atoms = self.num_basis*self.num_reps
        self.sc_pos = np.tile(self.basis.reshape(1,self.num_basis,3),reps=(self.num_reps,1,1))
        self.sc_types = np.tile(self.types,reps=(self.num_reps,1))

        _x, _y, _z = np.meshgrid(np.arange(self.reps[0]),
                                 np.arange(self.reps[1]),
                                 np.arange(self.reps[2]),indexing='ij')
        _x = _x.flatten(); _y = _y.flatten(); _z = _z.flatten()

        # integer translation vectors
        self.sc_shift = np.array((_x,_y,_z),dtype=int).T
        self.sc_shift.shape = [self.num_reps,1,3]
        self.sc_shift = np.tile(self.sc_shift,reps=(1,self.num_basis,1))

        # positions in reduced coords of supercell
        self.sc_pos += self.sc_shift
        self.sc_pos[:,:,0] /= self.reps[0]
        self.sc_pos[:,:,1] /= self.reps[1]
        self.sc_pos[:,:,2] /= self.reps[2]

        # probably should call this externally
        self.get_cartesian_coords()

    # ----------------------------------------------------------------------------------------------

    def get_cartesian_coords(self):

        """
        flatten supercell positions into num_atoms x 3 array and get in cartesian coords
        """

        self.sc_pos.shape = [self.num_atoms,3]
        self.sc_types.shape = [self.num_atoms]

        self.sc_cart = np.zeros(self.sc_pos.shape)
        for ii in range(self.num_atoms):
            self.sc_cart[ii,:] = self.sc_lattice_vectors[0,:]*self.sc_pos[ii,0]+ \
                                 self.sc_lattice_vectors[1,:]*self.sc_pos[ii,1]+ \
                                 self.sc_lattice_vectors[2,:]*self.sc_pos[ii,2]

    # ----------------------------------------------------------------------------------------------

    def write_poscar(self,file_name='POSCAR'):

        """
        write a VASP 'POSCAR' file for calculating/visualizing with VESTA
        """

        pos = self.sc_cart
        types = self.sc_types+1
        inds = np.argsort(types)

        num_ti = np.count_nonzero(types == 1)
        num_o = np.count_nonzero(types == 2)

        types = types[inds]
        pos = pos[inds]

        with open(file_name,'w') as f_out:
            _= 0.0
            f_out.write(f'auto generated\n 1.0\n')
            f_out.write(f'  {self.sc_lattice_vectors[0,0]:10.7f}  {_:10.7f}  {_:10.7f}\n')
            f_out.write(f'  {_:10.7f}  {self.sc_lattice_vectors[1,1]:10.7f}  {_:10.7f}\n')
            f_out.write(f'  {_:10.7f}  {_:10.7f}  {self.sc_lattice_vectors[2,2]:10.7f}\n')
            f_out.write(f' Ti O \n')
            f_out.write(f'   {num_ti:g}  {num_o:g}\nCartesian\n')
            for ii in range(self.num_atoms):
                f_out.write(f' {pos[ii,0]:10.9f}  {pos[ii,1]:10.9f}  {pos[ii,2]:10.9f}\n')

    # ----------------------------------------------------------------------------------------------

    def get_neighbors(self):

        """
        get 'neighbor' lists and distances using minimum image convention
        """

        _num_atoms = self.num_atoms
        _lat_vecs = self.sc_lattice_vectors
        _l0 = np.tile(_lat_vecs[0,:].reshape(1,3),reps=(_num_atoms,1))
        _l1 = np.tile(_lat_vecs[1,:].reshape(1,3),reps=(_num_atoms,1))
        _l2 = np.tile(_lat_vecs[2,:].reshape(1,3),reps=(_num_atoms,1))

        self.nn_list = np.zeros((_num_atoms,_num_atoms),dtype=int)
        self.nn_vecs = np.zeros((_num_atoms,_num_atoms,3),dtype=float)
        self.nn_cart = np.zeros((_num_atoms,_num_atoms,3),dtype=float)
        self.nn_dist = np.zeros((_num_atoms,_num_atoms),dtype=float)

        for ii in range(_num_atoms):

            # position of atom 'ii' in reduced coords
            _rii = self.sc_pos[ii,:].reshape(1,3)
            _rii = np.tile(_rii,reps=(_num_atoms,1))

            # ... in cartesian coords
            _cii = self.sc_cart[ii,:].reshape(1,3)
            _cii = np.tile(_cii,reps=(_num_atoms,1))

            # vector from atom 'ii' to all others in reduced coords
            _rrel = self.sc_pos-_rii

            # find atoms to shift in reduced coords
            _rshift = -(_rrel > 1/2).astype(int)
            _rshift += (_rrel <= -1/2).astype(int)

            # shift in both reduced and cartesian coords
            _cshift = np.tile(_rshift[:,0].reshape(_num_atoms,1),reps=(1,3))*_l0 + \
                      np.tile(_rshift[:,1].reshape(_num_atoms,1),reps=(1,3))*_l1 + \
                      np.tile(_rshift[:,2].reshape(_num_atoms,1),reps=(1,3))*_l2

            # relative vector in cartesian coords
            _crel = (self.sc_cart+_cshift)-_cii
            _rrel += _rshift
            
            # get sorted distance
            _d = np.sqrt(np.sum(_crel**2,axis=1))
            _inds = np.argsort(_d)

            self.nn_dist[ii,:] = _d[_inds]
            self.nn_list[ii,:] = _inds
            self.nn_vecs[ii,:,:] = _rrel[_inds,:]
            self.nn_cart[ii,:,:] = _crel[_inds,:]

    # ----------------------------------------------------------------------------------------------

    






