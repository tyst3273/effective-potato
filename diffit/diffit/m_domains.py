
import numpy as np
from diffit.m_code_utils import crash, c_timer
from diffit.m_crystal_utils import change_coordinate_basis, do_minimum_image



class c_domains:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,crystal):

        """
        class to select and modify 'domains' within a crystal 
        """
        
        self.crystal = crystal
        
    # ----------------------------------------------------------------------------------------------

    def find_slab(self,origin,vector,thickness,periodic=True):

        """
        find atoms within a 'slab'. origin is any point (in CARTESIAN COORDS) on the 'lower' 
        plane of the slab. 'vector' is the orientation vector (in CARTESIAN COORDS) perpendicular 
        to the slab. note, vector is enforced to be normalized. 'thickness' is the thickness 
        of the slab.
        """
        
        _t = c_timer('find_slab')
        
        origin = np.array(origin,dtype=float)
        vector = np.array(vector,dtype=float)
        vector = vector/np.sqrt(np.sum(vector**2))
        thickness = np.float(thickness)
        
        print('\n*** find slab ***\n')
        print('point on lower plane of slab:',origin)
        print('orientation vector of slab:',vector)
        print('thickness of slab:',thickness)
        
        # need origin in reduced coords since minimum image is done in reduced coords
        origin_reduced = change_coordinate_basis(self.crystal.sc_vectors_inv,origin)
        reduced_pos = np.copy(self.crystal.sc_positions_reduced)
        
        if periodic:
            reduced_pos = do_minimum_image(origin_reduced,reduced_pos)
        else:
            reduced_pos[:,0] += -origin_reduced[0]
            reduced_pos[:,1] += -origin_reduced[1]
            reduced_pos[:,2] += -origin_reduced[2]

        cart_pos = change_coordinate_basis(self.crystal.sc_vectors,reduced_pos)
                
        # get dot product of pos with vector
        vector = np.tile(vector.reshape(1,3),reps=(self.crystal.num_sc_atoms,1))
        _dot = np.sum(cart_pos*vector,axis=1)
        
        self.inds_in_slab = np.intersect1d(np.flatnonzero(_dot >= 0.0),
                                            np.flatnonzero(_dot <= thickness))
        
        if self.inds_in_slab.size == 0:
            print('\n!!! WARNING !!!')
            print('the slab is empty! continuing\n')
            
        _t.stop()
        
        return self.inds_in_slab
    
    # ----------------------------------------------------------------------------------------------

    def replace_slab_types(self,new_type,old_types=None):

        """
        appends a new 'type' to the types in the crystal and replaces all atoms of specified type
        'old_types' in the slab with the new type. if old_types is None, all types are replaced
        """
        
        _c = self.crystal
        
        if old_types is None:
            old_types = np.arange(_c.num_basis_types)
        if not isinstance(old_types,list):
            old_types = [old_types]
        
        _new_num = _c.add_new_basis_type(new_type)
        
        _nums = _c.sc_type_nums
        _str = _c.basis_type_strings
        for ii, t in enumerate(old_types):
            _t_ind = np.flatnonzero(_str == t)[0]
            _inds = np.flatnonzero(_nums == _t_ind)
            _inds = np.intersect1d(self.inds_in_slab,_inds)
            _nums[_inds] = _new_num
            
    # ----------------------------------------------------------------------------------------------

    def displace_slab(self,vector):

        """
        displace all of the atoms in the slab by 'vector'. vector is in CARTESIAN coordinats
        """
        
        _c = self.crystal
        _inds = self.inds_in_slab
        
        _c.sc_positions_cart[_inds,0] += vector[0]
        _c.sc_positions_cart[_inds,1] += vector[1]
        _c.sc_positions_cart[_inds,2] += vector[2]
            
    # ----------------------------------------------------------------------------------------------

    def get_crystal(self):

        """
        self explanatory
        """
        
        return self.crystal
    
    # ----------------------------------------------------------------------------------------------

 







    






