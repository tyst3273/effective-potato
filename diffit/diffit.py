
import numpy as np
import os

from PSF import c_PSF
from mods.m_crystals import c_rutile
from mods.m_lammps import c_lammps


# where to write/run the lammps calcs
lammps_top_dir = 'lammps_relax'
lammps_in = 'lammps_inputs/relax.in'
force_field = 'lammps_inputs/ffield.comb3'


# list 'c_rutile' objects that model instances of rutile
supercells = []

# target supercell size and steps around it to remove finite size effects
reps = [10,10,10]
num_sc = 5

# defect concentration (fraction or unitcells in sc)
num_defects = 10


# --------------------------------------------------------------------------------------------------
# set up supercells 

for ii in range(num_sc):
    
    rutile = c_rutile()
    rutile.make_supercell(reps=reps)

    rutile.make_oxygen_frenkels(num_defects=num_defects)
    rutile.write_lammps('lammps.pos',directory=f'{lammps_top_dir}/{ii:g}')

    supercells.append(rutile)
    print(f'{ii}\n')


# --------------------------------------------------------------------------------------------------
# go run the lammps calcs

# get the lammps directories to do
_ = [os.path.join(lammps_top_dir,d) for d in os.listdir(lammps_top_dir)]
dirs = []
for d in _:
    if os.path.isdir(d):
        dirs.append(d)

lammps = c_lammps(dirs)
lammps.setup_jobs(lammps_in,force_field)
lammps.setup_lammps(log_file='log.lammps')
lammps.run_lammps()


# --------------------------------------------------------------------------------------------------
# get the positions from the lammps calcs

for ii, sc in enumerate(supercells):
    
    d = os.path.join(lammps_top_dir,f'{ii:g}')
    pos_file = os.path.join(d,'positions.final')
    sc.get_pos_from_txt_file(pos_file)

    print(sc.sc_lattice_vectors[0,0]/sc.reps[0])
    print(sc.sc_lattice_vectors[1,1]/sc.reps[1])
    print(sc.sc_lattice_vectors[2,2]/sc.reps[2])


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
    







