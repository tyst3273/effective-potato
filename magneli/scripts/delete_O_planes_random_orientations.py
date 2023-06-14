
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_crystal_utils import get_neighbors_for_all_atoms_no_minimum_image
from diffit.m_code_utils import c_timer
from diffit.m_structure_io import write_xyz, write_poscar, write_lammps_data_file
from diffit.m_domains import c_embedded, c_domains

from diffit.m_PSF_interface import run_PSF

#rng = np.random.default_rng()
#i = rng.normal(loc=5,scale=1,size=500).round()


_t = c_timer('run_diffit',units='m')

# --------------------------------------------------------------------------------------------------

rutile = c_crystal(poscar='TiO2/POSCAR_TiO2_ideal')
a = rutile.basis_vectors[0,0]; c = rutile.basis_vectors[2,2]

#rutile.build_supercell([20,20,30])
rutile.build_supercell([10,10,20])


#write_poscar('POSCAR_rutile_supercell',rutile)

# --------------------------------------------------------------------------------------------------

# distances between O planes 
interplanar_distance = 1.0402232469225114 

# shear plane orientation vectors
orientation_vectors = np.array([[ 1,-3, 2],  # 0
                                [-1, 3, 2],  # 0
                                [ 3,-1, 2],  # 0
                                [-3, 1, 2],  # 0
                                [ 1, 3, 2],  # 1/2
                                [ 1, 3,-2],  # 1/2
                                [ 3, 1, 2],  # 1/2
                                [ 3, 1,-2]], # 1/2
                                dtype=float)
orientation_vectors[:,0] /= a; orientation_vectors[:,1] /= a; orientation_vectors[:,2] /= c

# shear planes pass thru this coordinate
origins = np.array([[ 1.14825, 1.14825, 0.00000],
                    [ 1.14825, 1.14825, 0.00000],
                    [ 1.14825, 1.14825, 0.00000],
                    [ 1.14825, 1.14825, 0.00000],
                    [ 1.14825, 3.44475, 1.49050],
                    [ 1.14825, 3.44475, 1.49050],
                    [ 1.14825, 3.44475, 1.49050],
                    [ 1.14825, 3.44475, 1.49050]])


# --------------------------------------------------------------------------------------------------

domains = c_domains(rutile)
vector_inds = np.arange(8)

epsilon = 0.05

num_max = 10
num_defects = 0

while True:

    np.random.shuffle(vector_inds)
    ind = vector_inds[0]

    step = np.random.randint(-100,101)

    vector = orientation_vectors[ind,:]
    origin = origins[ind,:]

    # we want it normalized
    vector *= 1/np.sqrt(vector@vector) 

    delta = vector*epsilon
    offset = origin+vector*step*interplanar_distance

    inds = domains.find_slab(offset-delta,vector,thickness=2*epsilon)
        
    nums = domains.crystal.sc_type_nums[inds]
    nums = np.flatnonzero(nums == 1)
    inds = inds[nums]

    if inds.size == 0:
        continue

    domains.merge_slab_inds(inds)
    domains.crystal.delete_atoms(inds)

    num_defects += 1
    if num_defects == num_max:
        print(f'\nadded {num_max} oxygen vacancy planes to the crystal!\n')
        break

# --------------------------------------------------------------------------------------------------        

rutile = domains.get_crystal()

#write_lammps_data_file('lammps.comb.in',rutile,atom_masses=[47.867,15.999],
#        atom_charges=[1.900,-0.950],tilted=True)

num_Ti = np.flatnonzero(rutile.sc_type_nums == 0).size
num_O = np.flatnonzero(rutile.sc_type_nums == 1).size

Z_Ti = 2.196 
Z = Z_Ti*num_Ti
Z_O = -Z/num_O

write_lammps_data_file('lammps.MA.in',rutile,atom_masses=[47.867,15.999],
        atom_charges=[Z_Ti,Z_O],tilted=True)

write_xyz('out.xyz',rutile)

_t.stop()




