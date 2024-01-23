
import numpy as np
from diffit.m_code_utils import crash, c_timer
from diffit.m_crystal_utils import change_coordinate_basis, do_minimum_image, \
            get_neighbors_for_all_atoms_no_minimum_image
            

# --------------------------------------------------------------------------------------------------

class c_domains:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,crystal):

        """
        class to select and modify 'domains' within a crystal 
        """
        
        self.crystal = crystal
        
    # ----------------------------------------------------------------------------------------------

    def merge_slab_inds(self,set_of_inds):

        """
        merge together sets of indices to form a new slab. useful to e.g. form a domain bounded 
        by a parelleliped or smth
        """

        if not isinstance(set_of_inds,list):
            set_of_inds = [set_of_inds]

        inds = np.array(set_of_inds[0],dtype=int)
        for ii in range(1,len(set_of_inds)):
            # inds = np.intersect1d(inds,np.array(set_of_inds[ii],dtype=int))
            inds = np.union1d(inds,np.array(set_of_inds[ii],dtype=int))
        
        self.inds_in_slab = inds

        return self.inds_in_slab

    # ----------------------------------------------------------------------------------------------

    def find_slab(self,origin,vector,thickness=1e6,periodic=False):

        """
        find atoms within a 'slab'. origin is any point (in CARTESIAN COORDS) on the 'lower' 
        plane of the slab. 'vector' is the orientation vector (in CARTESIAN COORDS) perpendicular 
        to the slab. note, vector is enforced to be normalized. 'thickness' is the thickness 
        of the slab.
        """
        
        _t = c_timer('find_slab')

        origin = np.array(origin,dtype=float)
        vector = np.array(vector,dtype=float)
        vector /= np.linalg.norm(vector)
        thickness = float(thickness)
        
        print('\n*** find slab ***\n')
        print('point on lower plane of slab:',origin)
        print('orientation vector of slab:',vector)
        print('thickness of slab:',thickness)
        
        if periodic:
     #       msg = 'find_slab(periodic=True) is no longer implemented!'
     #       crash(msg)
            # need origin in reduced coords since minimum image is done in reduced coords
            origin_reduced = change_coordinate_basis(self.crystal.sc_vectors_inv,origin)
            reduced_pos = np.copy(self.crystal.sc_positions_reduced)
            reduced_pos = do_minimum_image(origin_reduced,reduced_pos)
            cart_pos = change_coordinate_basis(self.crystal.sc_vectors,reduced_pos)
        else:
            cart_pos = np.copy(self.crystal.sc_positions_cart)
            cart_pos[:,0] += -origin[0]
            cart_pos[:,1] += -origin[1]
            cart_pos[:,2] += -origin[2]

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
            old_types = list(_c.basis_type_strings)
        if not isinstance(old_types,list):
            old_types = [old_types]

        for t in old_types:
            if t not in _c.basis_type_strings:
                msg = 'type {t} is unknown!'
                crash(msg)
        
        _new_num = _c.add_new_basis_type(new_type)
        
        _nums = _c.sc_type_nums
        _str = _c.basis_type_strings
        for ii, t in enumerate(old_types):
            _t_ind = np.flatnonzero(_str == t)[0]
            _inds = np.flatnonzero(_nums == _t_ind)
            _inds = np.intersect1d(self.inds_in_slab,_inds)
            _nums[_inds] = _new_num
            
    # ----------------------------------------------------------------------------------------------

    def delete_overlapping_atoms(self,cutoff=1e-3):

        """
        delete one of a pair of atoms that are closer together than 'cutoff' to eachother
        """
        
        _n, _d = get_neighbors_for_all_atoms_no_minimum_image(self.crystal)
        _n = _n[:,1]; _d = _d[:,1]

        print(_d.max(),_d.min())

        _ovlp = np.flatnonzero(_d <= cutoff)

        if _ovlp.size == 0:
            print('\nno overlapping atoms!\n')
        else:
            print(f'\nthere are {_ovlp.size} pairs of overlapping atoms!\n')
            self.crystal.delete_atoms(_ovlp)

    # ----------------------------------------------------------------------------------------------
    
    def displace_slab(self,vector,update_reduced=True):

        """
        displace all of the atoms in the slab by 'vector'. vector is in CARTESIAN coordinats

        can optionally not update reduced coord to save time but dont forget to do it later!
        """
        
        _c = self.crystal
        _inds = self.inds_in_slab
        
        _c.sc_positions_cart[_inds,0] += vector[0]
        _c.sc_positions_cart[_inds,1] += vector[1]
        _c.sc_positions_cart[_inds,2] += vector[2]

        if update_reduced:
            _c.update_reduced_coords()
        else:
            print('\n*** WARNING ***\n')
            msg = 'not update reduced coords! they wont be in sync with cartesian coords.\n'
            msg += 'dont forget to update them manually later!\n'
            print(msg)
            
    # ----------------------------------------------------------------------------------------------

    def get_crystal(self):

        """
        self explanatory
        """
        
        return self.crystal
    
    # ----------------------------------------------------------------------------------------------

    def crop_crystal(self,origin,vectors,epsilon=0.0,debug_atom_type=None):
        
        """
        crop the crystal. place the origin of domain spanned by vectors at 'origin' and delete all
        atoms that lie outside the domain
        """

        origin = np.array(origin,dtype=float)
        
        inds = []
        # get indices of atom in bulk crystal bounded by the unitcell of the defect structure
        vectors = np.array(vectors,dtype=float)

        for ii in range(3):
            vec = np.copy(vectors[ii,:])
            thickness = np.sqrt(np.sum(vec**2))
            vec /= thickness
            inds.append(self.find_slab(origin-vec*epsilon,vec,thickness+epsilon,periodic=False))

        inds = self.merge_slab_inds(inds)

        if debug_atom_type is not None:
            self.replace_slab_types('C')
            return

        # update the crystal
        _c = self.crystal
        _c.sc_vectors = vectors
        _c.sc_vectors_inv = np.linalg.inv(vectors)
        _c.sc_type_nums = _c.sc_type_nums[inds]
        _c.sc_positions_cart = _c.sc_positions_cart[inds,:]
        _c.sc_positions_cart[:,0] -= _c.sc_positions_cart[:,0].min()
        _c.sc_positions_cart[:,1] -= _c.sc_positions_cart[:,1].min()
        _c.sc_positions_cart[:,2] -= _c.sc_positions_cart[:,2].min()
        _c.num_sc_atoms = inds.size
        _c.update_reduced_coords()
        
    # ----------------------------------------------------------------------------------------------

 
