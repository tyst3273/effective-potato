
import numpy as np
import os

from PSF import c_PSF
from mods.m_crystals import c_rutile
from mods.m_lammps import c_lammps


# where to write/run the lammps calcs
lammps_top_dir = 'lammps_relax'
lammps_in = os.path.join(lammps_top_dir,'relax.in')
force_field = os.path.join(lammps_top_dir,'ffield.comb3')


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
exit()


# list 'c_rutile' objects that model instances of rutile
supercells = []

# target supercell size and steps around it to remove finite size effects
num_reps_target = [10,10,10]
d_reps = 5

# defect concentration (fraction or unitcells in sc)
concentration = 0.01


# --------------------------------------------------------------------------------------------------
# set up supercells 

_ = 0
for rr in range(-d_reps,d_reps+1):
    
    reps = np.array(num_reps_target)+rr

    rutile = c_rutile()
    rutile.make_supercell(reps=reps)

    rutile.make_oxygen_frenkels(concentration=concentration)
    rutile.write_lammps('lammps.pos',directory=f'{lammps_top_dir}/{_:g}')
    _ += 1

    supercells.append(rutile)
    print('')


# --------------------------------------------------------------------------------------------------
# go run the lammps calcs

dirs = [os.path.join(lammps_top_dir,d) for d in os.listdir(lammps_top_dir)]
lammps = c_lammps(dirs)
lammps.setup_jobs(lammps_in,force_field)


# --------------------------------------------------------------------------------------------------
# go and calc scattering intensity from each supercell

for ind, sc in enumerate(supercells):

    prefix = f'o{ind:g}'
    print(prefix)
    
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
    







