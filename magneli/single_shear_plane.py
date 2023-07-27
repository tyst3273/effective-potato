
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_crystal_utils import get_neighbors_for_all_atoms_no_minimum_image
from diffit.m_code_utils import c_timer
from diffit.m_structure_io import write_xyz, write_poscar
from diffit.m_domains import c_embedded, c_domains

from diffit.m_PSF_interface import run_PSF


_t = c_timer('run_diffit',units='m')

# --------------------------------------------------------------------------------------------------

rutile = c_crystal(poscar='TiO2/POSCAR_TiO2_ideal')
a = rutile.basis_vectors[0,0]; c = rutile.basis_vectors[2,2]

#rutile.build_supercell([20,20,30])
rutile.build_supercell([10,10,16])

write_poscar('POSCAR_rutile_supercell',rutile)

# --------------------------------------------------------------------------------------------------

# ascending distances between O planes (note, shear moves a
interplanar_distance = 1.22753088
origin_coords = np.arange(10)*interplanar_distance 


domains = c_domains(rutile)

# shear displacement vector
displacement_vector = np.array([0,a/2,c/2],dtype=float)

# shear plane orientation vector
orientation_vector = np.array([1/a,-3/a,2/c],dtype=float)

# we want it normalized
orientation_vector *= 1/np.sqrt(orientation_vector@orientation_vector) 

# a coordinate on the shear plane
origin = [ 1.14825, 1.14825, 0.00000]


slab_thickness = 5

domains.find_slab(origin,orientation_vector,slab_thickness)
domains.displace_slab(displacement_vector)
domains.delete_overlapping_atoms(cutoff=1e-3)



#rutile = domains.get_crystal()
#d, v = get_neighbors_for_all_atoms_no_minimum_image(rutile)

# print(d[:,1].max(),d[:,1].min())

#domains.find_slab(origin+orientation_vector*slab_thickness,orientation_vector)
#domains.replace_slab_types('Si')

#domains.find_slab(origin,orientation_vector)
#domains.replace_slab_types('C')
#domains.displace_slab(displacement_vector)

#domains.find_slab(origin+orientation_vector*slab_thickness,orientation_vector,slab_thickness)
#domains.replace_slab_types('Si')
#domains.displace_slab(displacement_vector)

# --------------------------------------------------------------------------------------------------

rutile = domains.get_crystal()

psf_kwargs = {'atom_types':['O','Ti'],
              'experiment_type':'neutrons',
              'num_Qpoint_procs':16,
              'Qpoints_option':'mesh',
              'Q_mesh_H':[0,4,41],
              'Q_mesh_K':[0,4,5],
              'Q_mesh_L':[0,4,65],
              'output_prefix':'magneli_single_defect'} 
run_PSF(rutile,**psf_kwargs)


# --------------------------------------------------------------------------------------------------

write_xyz('out.xyz',rutile)

# --------------------------------------------------------------------------------------------------

_t.stop()




