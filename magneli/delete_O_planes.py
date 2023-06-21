
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

nx = 20; nz = 30
rutile.build_supercell([nx,nx,nz])

#write_poscar('POSCAR_rutile_supercell',rutile)

# --------------------------------------------------------------------------------------------------

# ascending distances between O planes 
interplanar_distance = 1.0402232469225114 #1.22753088

# shear plane orientation vectors
orientation_vectors = np.loadtxt('vectors/CS_132_vectors')
orientation_vectors[:,0] /= a
orientation_vectors[:,1] /= a
orientation_vectors[:,2] /= c
num_vectors = orientation_vectors.shape[0]
for ii in range(num_vectors):
    orientation_vectors[ii,:] /= np.sqrt(orientation_vectors[ii,:]@orientation_vectors[ii,:])

# shear planes pass thru this coordinate
origins = np.loadtxt('vectors/CS_132_origins')
origins[:,0] *= a
origins[:,1] *= a
origins[:,2] *= c

# possible displacement vectors for the shear
displacement_vectors = np.loadtxt('vectors/CS_132_displacements')
displacement_vectors[:,0] *= a/2
displacement_vectors[:,1] *= a/2
displacement_vectors[:,2] *= c/2

# --------------------------------------------------------------------------------------------------

rng = np.random.default_rng()
normal = rng.normal(loc=12,scale=1,size=50).round().astype(int)

domains = c_domains(rutile)

epsilon = 0.05
radii = np.abs(rng.normal(loc=30,scale=5,size=5))

# orientation vectors 
for ii in [0]:

    vector = orientation_vectors[ii,:]
    delta = vector*epsilon
    origin = origins[ii,:]
    origin[0] += nx//2*a; origin[1] += nx//2*a; origin[2] += nz//2*c

    for jj in range(-2,3):

        np.random.shuffle(normal)
        n = normal[0]

        offset = n*jj*interplanar_distance*vector
        center = origin+offset

        delete = domains.find_slab(center-delta,vector,thickness=2*epsilon)
        if delete.size == 0:
            print('\nempty!\n')
            continue

        # find atoms within certain distance of center
        np.random.shuffle(radii)
        r = radii[0]
        pos = domains.crystal.sc_positions_cart[delete,:]
        pos[:,0] += -center[0]; pos[:,1] += -center[1]; pos[:,2] += -center[2]
        dist = np.sqrt(np.sum(pos**2,axis=1))
        inds = np.flatnonzero(dist <= r)
        delete = delete[inds]

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

# --------------------------------------------------------------------------------------------------        

rutile = domains.get_crystal()

num_Ti = np.flatnonzero(rutile.sc_type_nums == 0).size
num_O = np.flatnonzero(rutile.sc_type_nums == 1).size

Z_Ti = 2.196 
Z = Z_Ti*num_Ti
Z_O = -Z/num_O

write_lammps_data_file('lammps.MA.in',rutile,atom_masses=[47.867,15.999],
        atom_charges=[Z_Ti,Z_O],tilted=False)

write_xyz('out.xyz',rutile)

_t.stop()




