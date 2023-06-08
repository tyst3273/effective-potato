
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_code_utils import c_timer
from diffit.m_structure_io import write_xyz, write_lammpstrj, write_poscar
#from diffit.m_PSF_interface import run_PSF
#from diffit.m_rmc import c_rmc


_t = c_timer('run_diffit',units='m')

rutile = c_crystal(poscar='TiO2/POSCAR_TiO2')
magneli = c_crystal(poscar='Ti5O9/POSCAR_Ti5O9')

rutile.build_supercell()
write_poscar('POSCAR_rutile',rutile,cartesian=True)

magneli.build_supercell()
R = [[ 0.47638749,  0.76836498, -0.23168947],
     [-0.8200697,   0.6218852,  -0.30209542],
     [-0.11005654,  0.3689185,   0.92889442]]
magneli.rotate_coords(R)
write_poscar('POSCAR_mangeli_supercell',magneli,cartesian=True)

#ruile.build_supercell()
#write_poscar('POSCAR_rutile,rutile)

# setup supercells
#supercell_reps = [24,24,36]
#rutile.build_supercell(supercell_reps)

_t.stop()




