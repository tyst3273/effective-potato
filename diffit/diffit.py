
import numpy as np

from PSF import c_PSF
from mods.m_crystals import c_rutile



reps = [1,1,1]
rutile = c_rutile()
rutile.make_supercell(reps)
#rutile.write_poscar()

rutile._get_neighbors()
print(rutile.nn_dist)

exit()






# list 'c_rutile' objects that model instances of rutile
supercells = []

# target supercell size and steps around it to remove finite size effects
num_reps_target = [20,20,20]
d_reps = 5


# set up supercells from 'c_rutile' objects
for rr in range(-d_reps,d_reps+1):
    
    reps = np.array(num_reps_target)+rr

    rutile = c_rutile(basis=basis)
    rutile.make_supercell(reps=reps)

    supercells.append(rutile)



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
    




