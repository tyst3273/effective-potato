
import numpy as np
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_code_utils import c_timer
from diffit.m_point_defects import c_point_defects
from diffit.m_structure_io import write_xyz, write_lammpstrj


_t = c_timer('run_diffit',units='m')

# define rutile 
basis_vectors = [[4.593,0.000,0.000],
                 [0.000,4.593,0.000],
                 [0.000,0.000,2.981]]
basis_positions = [[0.0000000000000000,  0.0000000000000000,  0.0000000000000000],
                   [0.5000000000000000,  0.5000000000000000,  0.5000000000000000],
                   [0.1953400114833092,  0.8046599885166907,  0.5000000000000000],
                   [0.8046599885166907,  0.1953400114833092,  0.5000000000000000],
                   [0.3046599885166907,  0.3046599885166907,  0.0000000000000000],
                   [0.6953400114833093,  0.6953400114833093,  0.0000000000000000]]
basis_types = ['Ti','Ti','O','O','O','O']

rutile = c_crystal(basis_vectors,basis_positions,basis_types)


# setup supercell
#supercell_reps = [10,10,16]
supercell_reps = [4,4,6]
rutile.build_supercell(supercell_reps)

# number of defects
num_sc_atoms = rutile.num_sc_atoms 
defect_concentration = 0.05
num_defects = int(num_sc_atoms*defect_concentration)

num_defects = 1

vacancies = c_point_defects(rutile)
vacancies.place_random_defects(num_defects)


max_iter = 100
exit_flag = False

for ii in range(max_iter):

    _step_timer = c_timer(f'step[{ii}]')

    if exit_flag:
        break

    # write the file
    write_lammpstrj('vacancies.lammpstrj',rutile.sc_positions_cart,
                                          rutile.sc_type_nums,
                                          rutile.sc_vectors,
                                          append=True,sort_by_type=True)

    # randomly move a defect
    defect_ind, neighbor_ind = vacancies.move_defect()

    # write the file
    write_lammpstrj('vacancies.lammpstrj',rutile.sc_positions_cart,
                                          rutile.sc_type_nums,
                                          rutile.sc_vectors,
                                          append=True,sort_by_type=True)

    # unmove the defect
    vacancies.move_defect(neighbor_ind,defect_ind)

    _step_timer.stop()



_t.stop()

