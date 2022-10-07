

import numpy as np


class c_rutile:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,basis=None,types=None,lattice_vectors=None,charges=None):

        """
        define the primitive unitcell;

        WARNING: nothing is really error checked here
        """

        # types of atoms; 0 is Ti, 1 is oxygen
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
                                             [0.000,0.000,2.981]])
        else:
            self.lattice_vectors = np.array(lattice_vectors)
        self.inv_lattice_vectors = np.linalg.inv(self.lattice_vectors)

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
        self.num_reps = np.prod(self.reps) # i.e. number of unitcells in supercell

        self.sc_lattice_vectors = np.copy(self.lattice_vectors)
        self.sc_lattice_vectors[0,:] *= self.reps[0]
        self.sc_lattice_vectors[1,:] *= self.reps[1]
        self.sc_lattice_vectors[2,:] *= self.reps[2]
        self.sc_inv_lattice_vectors = np.linalg.inv(self.sc_lattice_vectors)

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

        # flatten into num_atoms x ... arrays
        self.sc_pos.shape = [self.num_atoms,3]
        self.sc_types.shape = [self.num_atoms]

        # probably should call this externally ...
        self.get_cartesian_coords()

    # ----------------------------------------------------------------------------------------------

    def get_reduced_coords(self):

        """
        get in reduced coords 
        """

        self.sc_pos = np.zeros(self.sc_cart.shape)
        for ii in range(self.num_atoms):
            self.sc_pos[ii,:] = self.sc_inv_lattice_vectors[0,:]*self.sc_cart[ii,0]+ \
                                self.sc_inv_lattice_vectors[1,:]*self.sc_cart[ii,1]+ \
                                self.sc_inv_lattice_vectors[2,:]*self.sc_cart[ii,2]

    # ----------------------------------------------------------------------------------------------

    def get_cartesian_coords(self):

        """
        get in cart coords
        """

        self.sc_cart = np.zeros(self.sc_pos.shape)
        for ii in range(self.num_atoms):
            self.sc_cart[ii,:] = self.sc_lattice_vectors[0,:]*self.sc_pos[ii,0]+ \
                                 self.sc_lattice_vectors[1,:]*self.sc_pos[ii,1]+ \
                                 self.sc_lattice_vectors[2,:]*self.sc_pos[ii,2]

    # ----------------------------------------------------------------------------------------------

    def write_poscar(self,file_name='POSCAR',cartesian=False):

        """
        write a VASP 'POSCAR' file for calculating/visualizing with VESTA
        """

        if cartesian:
            pos = self.sc_cart
            _str = 'Cartesian'
        else:
            pos = self.sc_pos
            _str = 'Direct'

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
            f_out.write(f'   {num_ti:g}  {num_o:g}\n{_str}\n')
            for ii in range(self.num_atoms):
                f_out.write(f' {pos[ii,0]:10.9f}  {pos[ii,1]:10.9f}  {pos[ii,2]:10.9f}\n')

    # ----------------------------------------------------------------------------------------------

    def get_neighbors(self):

        """
        get 'neighbor' lists and distances using minimum image convention

        NOTE: it will take some reworking elsewhere, but only getting the neighbor lists 
        for the randomly selected vacancy sites will be WAY faster than doing it for the 
        whole crystal
        """

        _nn_cut = 10

        _num_atoms = self.num_atoms
        _lat_vecs = self.sc_lattice_vectors
        _l0 = np.tile(_lat_vecs[0,:].reshape(1,3),reps=(_num_atoms,1))
        _l1 = np.tile(_lat_vecs[1,:].reshape(1,3),reps=(_num_atoms,1))
        _l2 = np.tile(_lat_vecs[2,:].reshape(1,3),reps=(_num_atoms,1))

        # running out of memory with NxN matrix ... only store up to nn_cut number of neighbors
        self.nn_list = np.zeros((_num_atoms,_nn_cut),dtype=int)
        self.nn_vecs = np.zeros((_num_atoms,_nn_cut,3),dtype=float)
        self.nn_cart = np.zeros((_num_atoms,_nn_cut,3),dtype=float)
        self.nn_dist = np.zeros((_num_atoms,_nn_cut),dtype=float)
        self.nn_types =  np.zeros((_num_atoms,_nn_cut),dtype=int)

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
            _inds = np.argsort(_d)[:_nn_cut]

            self.nn_dist[ii,:] = _d[_inds]
            self.nn_list[ii,:] = _inds
            self.nn_vecs[ii,:,:] = _rrel[_inds,:]
            self.nn_cart[ii,:,:] = _crel[_inds,:]
            self.nn_types[ii,:] = self.sc_types[_inds]

    # ----------------------------------------------------------------------------------------------

    def make_oxygen_frenkels(self,concentration=None,num_defects=None):

        """
        seed frenkel defects in supercells for relaxation and/or diffuse calculation

        NOTE: concentration is given as the FRACTION of UNITCELLS that have defects
        """

        from random import shuffle

        if not concentration is None and num_defects is None:
            num_defects = self.num_reps*concentration
        if num_defects is None and concentration is None:
            exit('\nmust give concentration or num_defects!\n')    

        num_defects = int(num_defects)
        if num_defects > self.num_reps:
            exit('\nnum_defects must not be greater than number of unitcells in supercell!\n')
        if num_defects <= 0:
            exit('\nnum_defects must not be 0!\n')

        print('num_reps',self.num_reps)
        print('num_defects',num_defects)

        # get neighbor lists 
        self.get_neighbors()

        # get 'pairs' of neibors for creating frenkels
        self._get_oxy_pairs(num_defects)

        # get coords of dimer centered on 'neighbor' site; its done in cartesian coords
        self._set_oxy_dimer_coords()

        # now get reduced coords for the defected sc
        self.get_reduced_coords()

    # ----------------------------------------------------------------------------------------------

    def _set_oxy_dimer_coords(self):

        """
        get coords for vacancy, neighbor pair as frenkel defect. move vacancy atom to 
        dimerize with neigboring oxy atom; dimer is centered on original site of neigbor
        but shifted perpendicular to Ti-O bond
        """

        _dimer_len = 1.3189666409731529 # bond length of oxy dimer in Angstrom
        _eps = 1e-3

        _num = self.vn_pairs.shape[0]

        for ii in range(_num):
            _v = self.vn_pairs[ii,0]
            _n = self.vn_pairs[ii,1]

            # need nearest Ti ** in-plane ** neighbor of neighbor O atom
            _ti = np.flatnonzero(self.nn_types[_n] == 0)

            # use neighbor vectors to find in-plane Ti neighbors
            _c = self.nn_cart[_n,:,:]
            _c = _c[_ti,:]
            _in_plane = np.flatnonzero(np.abs(_c[:,2]) < _eps)

            # they are sorted by dist. so lowest index is closest neighbor; 
            _nn_vec = _c[_in_plane[0]]

            # dimer vector is perpendicular to Ti-O bond and in the XY plane
            _dimer_vec = np.array([-1/(_nn_vec[1]+(_nn_vec[0]**2/_nn_vec[1])),
                                    1/(_nn_vec[0]+(_nn_vec[1]**2/_nn_vec[0])),
                                    0])

            # normalize it to be the 'known' dimer bond length (known from MD relaxation)
            _dimer_vec = _dimer_vec/np.sqrt(np.sum(_dimer_vec**2))

            # reset coords of vacancy atom and neighbor to make dimer
            self.sc_cart[_v] = self.sc_cart[_n]+_dimer_vec/2
            self.sc_cart[_n] = self.sc_cart[_n]-_dimer_vec/2

    # ----------------------------------------------------------------------------------------------

    def _get_oxy_pairs(self,num_defects):

        """
        defect 'site' is really one oxygen atom moved to dimerize with a neighbor
        so we need indices of vacancy site and neighbor to dimerize with, i.e. pairs of indices
        """

        from random import shuffle

        # vacancy, neighbor pairs
        self.vn_pairs = np.zeros((num_defects,2),dtype=int)

        # indices of O atoms
        _o_inds = list(np.flatnonzero(self.sc_types == 1))

        for ii in range(num_defects):

            shuffle(_o_inds)

            # vacancy index
            _vac = _o_inds[0]

            # neighboring O atoms
            _o = np.flatnonzero(self.nn_types[_vac,:] == 1)
            _nn = self.nn_list[_vac,:]
            _nn = _nn[_o]

            # neighbors already sorted by distance; get 1st nn
            _nn = _nn[1] # 0th ind is the vacancy atom, dist == 0

            self.vn_pairs[ii,:] = [_vac,_nn]

    # ----------------------------------------------------------------------------------------------
    
    def free_memory(self):

        """
        delete refs to huge neighbor lists that we dont want to keep anymore
        """

        import gc

        try:
            del self.nn_vecs, self.nn_dist, self.nn_cart, self.nn_types, self.nn_list
        except Exception as _ex:
            _msg = '\n*** WARNING ***\n' \
                   'tried deleting nn_vecs, nn_dist, nn_cart, nn_types, and nn_list but it\n' \
                   'didnt work! do they exist? here is the exception:'
            print(_msg)
            print(_ex)
            print('')

        gc.collect()

    # ----------------------------------------------------------------------------------------------




    






