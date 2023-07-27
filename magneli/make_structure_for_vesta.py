
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_crystal_utils import get_neighbors_for_all_atoms_no_minimum_image
from diffit.m_code_utils import c_timer
from diffit.m_structure_io import write_xyz, write_poscar, write_lammps_data_file
from diffit.m_domains import c_domains

from diffit.m_PSF_interface import run_PSF


_t = c_timer('run_diffit',units='m')

# --------------------------------------------------------------------------------------------------

rutile = c_crystal(poscar='TiO2/POSCAR_TiO2_ideal')
a = rutile.basis_vectors[0,0]; c = rutile.basis_vectors[2,2]

nx = 12; nz = 18
rutile.build_supercell([nx,nx,nz])

#write_poscar('POSCAR_rutile_supercell',rutile)

# --------------------------------------------------------------------------------------------------

# ascending distances between O planes 
interplanar_distance = 1.0402232469225114 #1.22753088

# shear plane orientation vector
orientation_vector = np.loadtxt('vectors/CS_132_vectors')[0,:]
orientation_vector[0] /= a # reciprocal space
orientation_vector[1] /= a
orientation_vector[2] /= c
orientation_vector /= np.linalg.norm(orientation_vector)

# shear planes pass thru this coordinate
origin = np.loadtxt('vectors/CS_132_origins')[0,:]
origin[0] *= a # real space
origin[1] *= a
origin[2] *= c

# possible displacement vectors for the shear
displacement_vector = np.loadtxt('vectors/CS_132_displacements')[0,:]
displacement_vector[0] *= a/2
displacement_vector[1] *= a/2
displacement_vector[2] *= c/2

# --------------------------------------------------------------------------------------------------

domains = c_domains(rutile)

# pick center of supercell
origin[0] += nx//2*a
origin[1] += nx//2*a
origin[2] += nz//2*c

# find plane of O atoms to delete
epsilon = 0.05
delta = orientation_vector*epsilon
delete = domains.find_slab(origin-delta,orientation_vector,thickness=2*epsilon)
if delete.size == 0:
    msg = 'no atoms found in plane...'
    crash(msg)

# check that we didnt find any Ti atoms
nums = domains.crystal.sc_type_nums[delete]
if np.any(nums == 0):
    msg = 'found Ti atoms in the plane to delete...'
    crash(msg)

# delete the atoms
print(f'\ndeleting {delete.size} O atoms!\n')
domains.crystal.delete_atoms(delete)
#domains.merge_slab_inds(delete)
#domains.replace_slab_types('C')

# change types of Ti atoms near interface
epsilon = 1.00
delta = orientation_vector*epsilon
change = domains.find_slab(origin-delta,orientation_vector,thickness=2*epsilon)
nums = domains.crystal.sc_type_nums[change]
change = change[np.flatnonzero(nums == 0)]
domains.merge_slab_inds(change)
domains.replace_slab_types('Cu')


# displace the slab
domains.find_slab(origin,orientation_vector)
domains.displace_slab(displacement_vector)
#domains.delete_overlapping_atoms(cutoff=0.1)

"""
# define domain of atoms to keep
crop_vectors = np.copy(rutile.basis_vectors)
crop_vectors[0,0] *= nx-4; crop_vectors[1,1] *= nx-4; crop_vectors[2,2] *= nz-4
crop_origin = np.array([2*a,2*a,2*c])
domains.crop_crystal(crop_origin,crop_vectors,epsilon=0.05) #,debug_atom_type='Si')
"""

# --------------------------------------------------------------------------------------------------        

rutile = domains.get_crystal()

write_poscar('CS_132_POSCAR',rutile,cartesian=True)
write_xyz('out.xyz',rutile)

_t.stop()






