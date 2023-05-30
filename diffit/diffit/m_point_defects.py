
import numpy as np
from diffit.m_code_utils import crash, c_timer
from diffit.m_crystal_utils import change_coordinate_basis, do_minimum_image, get_neighbors



class c_point_defects:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,crystal):

        """
        class to select and modify which atoms are defects in the crystal
        """
        
        self.crystal = crystal
        self.sc_atom_inds = np.arange(self.crystal.num_sc_atoms)
        
    # ----------------------------------------------------------------------------------------------

    def get_random_inds(self,inds,num_inds=None):

        """
        randomly chose num_inds inds from sc_atom_inds
        """

        np.random.shuffle(inds)
        if num_inds is None:
            return inds
        else:
            return inds[:num_inds]

    # ----------------------------------------------------------------------------------------------

    def place_random_defects(self,num_defects,defect_type='vacancy'):

        """
        randomly place defects inside the crystal
        """

        self.defect_type = defect_type

        self.defect_sc_inds = self.get_random_inds(np.arange(self.crystal.num_sc_atoms))

        self.pristine_sc_inds = self.defect_sc_inds[num_defects:]
        self.num_pristine = self.pristine_sc_inds.size

        self.defect_sc_inds = self.defect_sc_inds[:num_defects]
        self.num_defects = num_defects

        print('\n*** random defects **\n')
        print('num_defects:',num_defects)
        print('num_pristine:',self.num_pristine)
        print('defect_type:',defect_type)

        self.pristine_sc_type_nums = np.copy(self.crystal.sc_type_nums)
        self._change_types(self.defect_sc_inds,defect_type)

    # ----------------------------------------------------------------------------------------------

    def _change_types(self,defect_sc_inds,defect_type):
        
        """
        change the type of the atoms at defect_sc_inds to defect_type
        """

        type_num = self.crystal.add_new_basis_type(defect_type)
        self.crystal.sc_type_nums[defect_sc_inds] = type_num

    # ----------------------------------------------------------------------------------------------

    def move_defect(self,defect_ind=None):

        """
        swap position of a randomly chosen defect atom with nearby pristine one
        """

        if defect_ind is None:
            defect_ind = self.get_random_inds(np.copy(self.defect_sc_inds))[0]

        # need neighbor vectors in cartesian coordinates
        pristine_reduced_pos = self.crystal.sc_positions_reduced[self.pristine_sc_inds]
        defect_reduced_pos = self.crystal.sc_positions_reduced[defect_ind]
        neighbors, neighbor_dist, neighbor_vectors = get_neighbors(defect_reduced_pos,
                                                pristine_reduced_pos,self.crystal.sc_vectors)
        neighbors = self.pristine_sc_inds[neighbors]

        # randomly pick a neighbor
        shell = np.flatnonzero(neighbor_dist <= neighbor_dist[0]+0.1)
        np.random.shuffle(shell)
        neighbor_ind = neighbors[shell[0]]

        self.swap_pristine_and_defect_atoms(neighbor_ind,defect_ind)

        return neighbor_ind, defect_ind

    # ----------------------------------------------------------------------------------------------

    def swap_pristine_and_defect_atoms(self,pristine_ind,defect_ind):
        
        """
        swap an original (pristine) atom and a defect atom
        """

        # type of pristine atom
        pristine_type = self.crystal.sc_type_nums[pristine_ind]
        pristine_type = self.crystal.basis_type_strings[pristine_type]
        
        # type of defect
        defect_type = self.crystal.sc_type_nums[defect_ind]
        defect_type = self.crystal.basis_type_strings[defect_type]

        # change their types in c_crystal
        self._change_types(pristine_ind,defect_type)
        self._change_types(defect_ind,pristine_type)

        # need to swap indices in defect and neighbor arrays too
        _n = np.flatnonzero(self.pristine_sc_inds == pristine_ind)[0]
        self.pristine_sc_inds[_n] = defect_ind
        self.pristine_sc_inds = np.sort(self.pristine_sc_inds)

        _d = np.flatnonzero(self.defect_sc_inds == defect_ind)[0]
        self.defect_sc_inds[_d] = pristine_ind
        self.defect_sc_inds = np.sort(self.defect_sc_inds)

    # ----------------------------------------------------------------------------------------------

    def get_crystal(self):

        """
        self explanatory
        """
        
        return self.crystal
    
    # ----------------------------------------------------------------------------------------------

 







    






