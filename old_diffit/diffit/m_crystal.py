
import numpy as np
from diffit.m_code_utils import crash, c_timer
from diffit.m_crystal_utils import change_coordinate_basis



class c_crystal:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,basis_vectors,basis_positions_reduced,basis_types,label='crystal'):
            
        """
        take user args for lattice vectors, atom positions_reduced, and types and set up
        a crystal object
        """
        
        # a 'label' that describes the structure
        self.label = str(label)
        
        self.basis_vectors = basis_vectors
        self.basis_positions_reduced = basis_positions_reduced
        self.basis_types = basis_types
        
        self._setup_crystal()
    
    # ----------------------------------------------------------------------------------------------

    def _setup_crystal(self):

        """
        parse user args to setup crystal
        """
        
        # 'lattice' vectors or primitive unitcell
        err = 'basis vectors seems wrong!'
        try:
            self.basis_vectors = np.array(self.basis_vectors,dtype=float)
            self.basis_vectors_inv = np.linalg.inv(self.basis_vectors)
        except Exception as ex:
            crash(err,ex)
        if self.basis_vectors.ndim != 2: 
            crash(err)
        if self.basis_vectors.shape[0] != 3 or self.basis_vectors.shape[1] != 3:
            crash(err)
            
        # positions of basis atoms in reduced coords
        err = 'basis_positions_reduced seems wrong!'
        try:
            self.basis_positions_reduced = np.array(self.basis_positions_reduced,dtype=float)
        except Exception as ex:
            crash(err,ex)
        if self.basis_positions_reduced.ndim != 2: 
            crash(err)
        if self.basis_positions_reduced.shape[1] != 3:
            crash(err)
        
        self.num_basis_atoms = self.basis_positions_reduced.shape[0]
    
        # convert str types to numbers and get inverse map
        err = 'basis_types seems wrong!'
        try:
            self.basis_types = np.array(self.basis_types,dtype=object)
        except Exception as ex:
            crash(err,ex)
        if self.basis_types.ndim != 1: 
            crash(err)
        if self.basis_types.size != self.num_basis_atoms:
            crash(err)
            
        self.basis_inds = np.arange(self.num_basis_atoms)
        self.basis_type_strings, self.basis_type_nums = \
                np.unique(self.basis_types,return_inverse=True)
        self.num_basis_types = self.basis_type_strings.size
        self.basis_type_string_inds = np.arange(self.num_basis_types)
                
        # get basis positions in cartesian coords
        self.basis_positions_cart = \
            change_coordinate_basis(self.basis_vectors,self.basis_positions_reduced)
    
        print('\n*** primitive cell ***\n')
        print('basis vectors:\n',self.basis_vectors)
        print('basis positions (reduced coords):\n',self.basis_positions_reduced)
        print('basis positions (cartesian coords):\n',self.basis_positions_cart)
        print('basis type strings:',self.basis_type_strings)
        print('basis type map:',self.basis_type_nums)
        
    # ----------------------------------------------------------------------------------------------
    
    def add_new_basis_type(self,new_type):
        
        """
        adds a new basis type (new_type is a string). if type already exists, returns the index
        """

        exists = (new_type in self.basis_type_strings)
        if exists:
            return np.flatnonzero(self.basis_type_strings == new_type)[0]
        
        new_type = np.array(new_type,dtype=object)
        self.basis_type_strings = np.r_[self.basis_type_strings,new_type]
        self.num_basis_types += 1
        self.basis_type_string_inds = np.arange(self.num_basis_types)
        
        new_basis_ind = self.num_basis_types-1
        return new_basis_ind

    # ----------------------------------------------------------------------------------------------

    def build_supercell(self,supercell_reps=[1,1,1]):

        """
        create rectangular supercell by replicating primitive cell n_ii number of times 
        along the ii^th direction 
        """

        _t = c_timer('build_supercell')

        err = 'supercell_reps seems wrong!'
        try:
            self.sc_reps = np.array(supercell_reps,dtype=int)
        except Exception as ex:
            crash(err,ex)
        if self.sc_reps.ndim != 1 or self.sc_reps.size != 3:
            crash(err)
            
        self.num_sc_reps = np.prod(self.sc_reps) # i.e. number of unitcells in sc
        self.num_sc_atoms = self.num_basis_atoms*self.num_sc_reps

        # supercell 'lattice' vectors
        self.sc_vectors = np.copy(self.basis_vectors)
        self.sc_vectors[0,:] *= self.sc_reps[0]
        self.sc_vectors[1,:] *= self.sc_reps[1]
        self.sc_vectors[2,:] *= self.sc_reps[2]
        
        # inverse of lattice vectors are needed to go from cartesian to reduced coords
        self.sc_vectors_inv = np.linalg.inv(self.sc_vectors)
        
        print('\n*** super cell ***\n')
        print('supercell replications:',self.sc_reps)
        print('number of unitcells in supercell:',self.num_sc_reps)
        print('number of atoms in supercell:',self.num_sc_atoms)
        print('supercell vectors:\n',self.sc_vectors)

        # copy basis positions to supercell
        self.sc_positions_reduced = np.tile(
                self.basis_positions_reduced.reshape(1,self.num_basis_atoms,3),
                reps=(self.num_sc_reps,1,1))
        self.sc_type_nums = np.tile(self.basis_type_nums.reshape(1,self.num_basis_atoms),
                reps=(self.num_sc_reps,1))

        # integer translation vectors
        _x, _y, _z = np.meshgrid(np.arange(self.sc_reps[0]),
                                  np.arange(self.sc_reps[1]),
                                  np.arange(self.sc_reps[2]),indexing='ij')
        _x = _x.flatten(); _y = _y.flatten(); _z = _z.flatten()
        self.sc_translations = np.array((_x,_y,_z),dtype=int).T
        self.sc_translations.shape = [self.num_sc_reps,1,3]
        self.sc_translations = np.tile(self.sc_translations,reps=(1,self.num_basis_atoms,1))

        # positions_reduced in reduced coords of supercell
        self.sc_positions_reduced += self.sc_translations
        self.sc_positions_reduced[:,:,0] /= self.sc_reps[0]
        self.sc_positions_reduced[:,:,1] /= self.sc_reps[1]
        self.sc_positions_reduced[:,:,2] /= self.sc_reps[2]

        # flatten into num_sc_atoms x ... arrays
        self.sc_positions_reduced.shape = [self.num_sc_atoms,3]
        self.sc_type_nums.shape = [self.num_sc_atoms]
        self.sc_translations.shape = [self.num_sc_atoms,3]

        # get sc positions in cartesian coords
        self.sc_positions_cart = \
            change_coordinate_basis(self.sc_vectors,self.sc_positions_reduced)

        _t.stop()
    
    # ----------------------------------------------------------------------------------------------
    






