
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_code_utils import c_timer
from diffit.m_structure_io import write_xyz, write_poscar
from diffit.m_domains import c_embedded
#from diffit.m_PSF_interface import run_PSF
#from diffit.m_rmc import c_rmc


_t = c_timer('run_diffit',units='m')

# --------------------------------------------------------------------------------------------------

rutile = c_crystal(poscar='TiO2/POSCAR_TiO2')
magneli = c_crystal(poscar='Ti5O9/POSCAR_magneli_rotated')

#rutile.build_supercell([20,20,30])
rutile.build_supercell([10,10,15])
magneli.build_supercell([1,8,8])

write_poscar('POSCAR',magneli)

# --------------------------------------------------------------------------------------------------

embedded = c_embedded(rutile,magneli)

# embed defect with no transformations
embedded.embed([10,10,10])

"""
# rotate defect by 90 and embed rotated version
matrix = [[ 0,-1, 0],
          [ 1, 0, 0],
          [ 0, 0, 1]]
embedded.transform_defect(matrix=matrix)
embedded.embed([100,50,50],atom_type='C')

# rotate defect by 90 and embed rotated version
matrix = [[ 0,-1, 0],
          [ 1, 0, 0],
          [ 0, 0, 1]]
embedded.transform_defect(matrix=matrix)
embedded.embed([150,50,50],atom_type='N')

# rotate defect by 90 and embed rotated version
matrix = [[ 0,-1, 0],
          [ 1, 0, 0],
          [ 0, 0, 1]]
embedded.transform_defect(matrix=matrix)
embedded.embed([200,50,50],atom_type='F')
"""

# --------------------------------------------------------------------------------------------------

rutile = embedded.get_crystal()
write_xyz('out.xyz',rutile)

# --------------------------------------------------------------------------------------------------

_t.stop()




