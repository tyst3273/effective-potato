
import numpy as np
from m_md import c_md

# define simulation
num_atoms = 100
pos = np.arange(num_atoms)
box_size = num_atoms
atom_types = np.ones(num_atoms)
masses = [1]

md = c_md(pos,box_size,atom_types,masses,epsilon=1,sigma=1,cutoff=30)

#calc_fc(md)
#check_force_cutoff(md)

dt = 0.001
damp = 0.01
temp = 0.1

md.set_velocities(temp)
#md.read_restart('restart.hdf5')

#md.run_nve(dt=dt,num_steps=100000)

md.run_nvt_nose_hoover(dt=dt,num_steps=100000,T=temp,damp=damp,hdf5_file='equil.hdf5')
md.run_nvt_nose_hoover(dt=dt,num_steps=100000,T=temp,damp=damp,hdf5_file='run.hdf5')

