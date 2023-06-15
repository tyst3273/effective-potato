
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_crystal_utils import get_neighbors_for_all_atoms_no_minimum_image
from diffit.m_code_utils import c_timer, crash
from diffit.m_structure_io import write_xyz, write_poscar, write_lammps_data_file
from diffit.m_domains import c_embedded, c_domains

from diffit.m_PSF_interface import run_PSF


_t = c_timer('run_diffit',units='m')

n = 5


# --------------------------------------------------------------------------------------------------

rutile = c_crystal(poscar='TiO2/POSCAR_TiO2_ideal')
a = rutile.basis_vectors[0,0]; c = rutile.basis_vectors[2,2]

nx = 20; nz = 30
rutile.build_supercell([nx,nx,nz])

# primitive tetragona symm ops 
rotations = np.loadtxt('rotations').reshape(16,3,3)

# distances between O planes 
interplanar_distance = 1.0402232469225114 

# shear plane orientation vectors
orientation_vector = np.array([  1/a, -3/a,  2/c])
orientation_vector /= np.sqrt(orientation_vector@orientation_vector)

# shear planes pass thru this coordinate
origin = np.array([  a/4,  a/4,    0])

# possible displacement vectors for the shear
displacement_vector = np.array([    0,  a/2,  c/2])

# --------------------------------------------------------------------------------------------------

domains = c_domains(rutile)

inds_to_displace = []
good_steps = []

epsilon = 0.005
delta = orientation_vector*epsilon


# delete planes of O atoms 
for step in range(-20,20):

    offset = origin+n*step*orientation_vector*interplanar_distance

    # get the O atoms
    delete = domains.find_slab(offset-delta,orientation_vector,2*epsilon)
    if delete.size == 0:
        print('\nempty!\n')
        continue

    good_steps.append(step)

    # check that we didnt find any Ti atoms
    nums = domains.crystal.sc_type_nums[delete]
    if np.any(nums == 0):
        msg = 'found Ti atoms in the plane to delete...'
        crash(msg)

    # delete the atoms
    print(f'\ndeleting {delete.size} O atoms!\n')
    domains.crystal.delete_atoms(delete)

nums = domains.crystal.sc_type_nums
num_Ti = np.count_nonzero(nums == 0)
num_O = np.count_nonzero(nums == 1)
gcd = np.gcd(num_Ti,num_O)
print(f'\nTi_{num_Ti//gcd}O_{num_O//gcd}\n')

# get the sets of inds to displace (have to do after deleting O atoms ...)
for step in good_steps:

    offset = origin+n*step*orientation_vector*interplanar_distance

    inds = domains.find_slab(offset,orientation_vector)
    inds_to_displace.append(inds)

# now go and displace the slabs
for inds in inds_to_displace:
    
    domains.merge_slab_inds(inds)
    domains.displace_slab(displacement_vector)



rutile = domains.get_crystal()

# --------------------------------------------------------------------------------------------------        

write_xyz('out.xyz',rutile)

psf_kwargs = {'atom_types':['Ti','O'],
              'experiment_type':'neutrons',
              'num_Qpoint_procs':16,
              'Qpoints_option':'mesh',
              'Q_mesh_H':[ 0.0, 3.0, nx*3+1],
              'Q_mesh_K':[ 0.0, 0.0, 1],
              'Q_mesh_L':[ 0.0, 3.0, nz*3+1],
              'output_prefix':'magneli'}
run_PSF(rutile,**psf_kwargs)


_t.stop()




