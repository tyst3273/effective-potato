
import numpy as np
import os

from PSF import c_PSF
from mods.m_crystals import c_rutile
from mods.m_lammps import c_lammps


# list 'c_rutile' objects that model instances of rutile
supercells = []

# target supercell size and steps around it to remove finite size effects
reps = [20,20,20]
num_sc = 10

# --------------------------------------------------------------------------------------------------
# set up supercells 

for ii in range(num_sc):
    
    rutile = c_rutile()
    rutile.make_supercell(reps=reps)
    rutile.make_oxygen_frenkels(num_defects=25)

    supercells.append(rutile)
    print(f'{ii}\n')


# --------------------------------------------------------------------------------------------------
# go and calc scattering intensity from each supercell

for ind, sc in enumerate(supercells):

    prefix = f'o{ind:g}'
    
    # object to calculate scattering intensity each time
    PSF = c_PSF(input_file='batch_config.py')

    # now overwrite all config variables we want to change
    PSF.config.set_config(md_num_atoms=sc.num_atoms,
                          md_supercell_reps=sc.reps,
                          output_prefix=prefix)

    # set stuff up
    PSF.setup_communicator(sc.sc_cart,
                           sc.sc_types)

    # get structure factors
    PSF.run()

    # write the results
    PSF.write_strufacs()
    







