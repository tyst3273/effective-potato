
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from diffit.m_crystal import c_crystal
from diffit.m_code_utils import c_timer
from diffit.m_point_defects import c_point_defects
from diffit.m_structure_io import write_xyz, write_lammpstrj, write_poscar
from diffit.m_PSF_interface import run_PSF
from diffit.m_rmc import c_rmc

from diffit.m_custom import put_into_bins, get_inds_and_weights


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
basis_types = ['Ti','Zr','O','O','O','O']
rutile = c_crystal(basis_vectors,basis_positions,basis_types)

rutile.build_supercell()

write_poscar('POSCAR',rutile)

exit()

# setup supercell
supercell_reps = [24,24,36]
rutile.build_supercell(supercell_reps)

# number of defects
defect_concentration = 0.01
num_defects = int(rutile.num_sc_atoms*defect_concentration)


# class to work with the point defects
vacancies = c_point_defects(rutile)
vacancies.place_random_defects(num_defects) # seed random defects


psf_kwargs = {'Q_file':'Q_H3.00_K1.00_L0.00'}


inds_file = 'k1_u_binning.hdf5'
inds, weights, bin_centers = get_inds_and_weights(inds_file)
exp_intensity = np.loadtxt('quenched_310_signal')

scale = 1e3
exp_intensity *= scale/exp_intensity.max()


# RMC loop
max_iter = 100
rmc = c_rmc(beta=1e-6,exit_tol=1e-9)

for ii in range(max_iter):

    _step_timer = c_timer(f'step[{ii}]')

    # randomly move a defect; note, this works on a reference to  the 
    # rutile object so any changes made in vacancies change rutile class
    defect_ind, neighbor_ind = vacancies.move_defect()

    # run the PSF calculation
    calc_intensity = run_PSF(rutile,**psf_kwargs)

    # need to put intensity into bins
    calc_intensity = put_into_bins(calc_intensity,inds,weights)
    calc_intensity *= scale/calc_intensity.max()

    #plt.plot(bin_centers,calc_intensity,c='r',marker='o',lw=1,ls='-')
    #plt.plot(bin_centers,exp_intensity,c='b',marker='o',lw=1,ls='-')
    #plt.show()
    #plt.clf()
    #plt.close()

    # check agreement with exp. data for RMC
    keep, converged = rmc.check_move(exp_intensity,calc_intensity)

    print('keep move:',keep)
    print('error**2:',rmc.error_squared)
    print('delta**2:',rmc.delta_error_squared)

    # check if converged
    if converged:
        print(f'rmc loop converged after {ii+1} steps!')
        print('final delta**2:',rmc.delta_error_squared)
        break

    # unmove the defect
    if not keep:
        vacancies.move_defect(neighbor_ind,defect_ind)

    # optionally write atom coords each step
    if False:
        write_lammpstrj('vacancies.lammpstrj',rutile.sc_positions_cart,
                                              rutile.sc_type_nums,
                                              rutile.sc_vectors,
                                              append=True,sort_by_type=True)

    _step_timer.stop()

# check if converged
if not converged:
    print(f'rmc loop failed to converge after {ii+1} steps!')
    print('final delta**2:',rmc.delta_error_squared)

_t.stop()

