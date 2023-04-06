
import numpy as np
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_code_utils import c_timer
from diffit.m_domains import c_domains
from diffit.m_point_defects import c_point_defects
from diffit.m_structure_io import write_xyz, write_lammpstrj


_t = c_timer('run_diffit',units='m')

# define silicon 
basis_vectors = [[5.431,0.000,0.000],
                 [0.000,5.431,0.000],
                 [0.000,0.000,5.431]]
basis_positions = [[ 0.00, 0.00, 0.00],
                   [ 0.50, 0.50, 0.00],
                   [ 0.50, 0.00, 0.50],
                   [ 0.00, 0.50, 0.50],
                   [ 0.25, 0.25, 0.25],
                   [ 0.75, 0.75, 0.25],
                   [ 0.75, 0.25, 0.75],
                   [ 0.25, 0.75, 0.75]]

basis_types = ['Si','Si','Si','Si', 'Si','Si','Si','Si']
silicon = c_crystal(basis_vectors,basis_positions,basis_types)


# setup supercell
supercell_reps = [10,10,10]
silicon.build_supercell(supercell_reps)


# create random distribution of vacancies
defects = c_point_defects(silicon)

defect_concentration = 0.2
num_defects = int(silicon.num_sc_atoms*defect_concentration)

defects.place_random_defects(num_defects,defect_type='Ge')


for ii in range(100):

    print(ii)

    defects.move_defect()
    silicon = defects.get_crystal()
    write_lammpstrj('Ge_subs.lammpstrj',
                        silicon.sc_positions_cart,
                        silicon.sc_type_nums,
                        silicon.sc_vectors,
                        append=True,sort_by_type=True)


_t.stop()

