
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

rutile.build_supercell([40,40,60])
#rutile.build_supercell([10,10,10])


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

displacement_vectors = np.array([[    0,  a/2,  c/2],
                                 [    0, -a/2,  c/2],
                                 [ -a/2,    0, -c/2],
                                 [  a/2,    0, -c/2],
                                 [    0, -a/2,  c/2],
                                 [    0, -a/2, -c/2],
                                 [ -a/2,    0, -c/2],
                                 [ -a/2,    0,  c/2]],
                                 dtype=float)

# --------------------------------------------------------------------------------------------------

domains = c_domains(rutile)
vector_inds = np.arange(8)

num_max = 10
num_defects = 0

while True:

    np.random.shuffle(vector_inds)
    ind = vector_inds[0]

    step = np.random.randint(-200,200)

    vector = orientation_vectors[ind,:]
    vector *= 1/np.sqrt(vector@vector)    

    origin = origins[ind,:]+vector*step*interplanar_distance

    inds = domains.find_slab(origin,vector)
    if inds.size == 0:
        continue

    disp = displacement_vectors[ind,:]
    domains.displace_slab(disp)

    num_defects += 1
    if num_defects == num_max:
        print(f'\nadded {num_max} oxygen vacancy planes to the crystal!\n')
        break

#domains.delete_overlapping_atoms()


# --------------------------------------------------------------------------------------------------        

rutile = domains.get_crystal()


psf_kwargs = {'atom_types':['Ti','O'],
              'experiment_type':'neutrons',
              'num_Qpoint_procs':16,
              'Qpoints_option':'mesh',
              'Q_mesh_H':[ 2.5, 3.5, 41],
              'Q_mesh_K':[-0.5, 0.5, 41],
              'Q_mesh_L':[ 0.5, 1.5, 61],
              'output_prefix':'magneli_single_defect'}
run_PSF(rutile,**psf_kwargs)


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




