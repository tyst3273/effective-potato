

# options for where to get data/preprocessing
trajectory_format = 'lammps_hdf5' 
trajectory_file = 'pos.h5'

unwrap_trajectory = True
calc_sqw = True

# options for splitting up trajectory
num_trajectory_blocks = 10
trajectory_blocks = [0,2]

# options for writing results
output_directory = None
output_prefix = 'psf'

# simulation inputs
md_time_step = 16     # femtoseconds; time step IN FILE, not Verlet time step
md_num_steps = 500    # number of steps IN FILE
md_num_atoms = 71276

# unit cell used to define Q-points in cartesian coords
lattice_vectors = [[ 4.492, 0.000, 0.000], # angstroms
                   [ 0.000, 4.492, 0.000],
                   [ 0.000, 0.000, 3.031]]

# vectors spanning the simulation cell
box_vectors = None

# experiment info
experiment_type = 'neutrons' # 'neutrons' or 'xrays'
atom_types = ['Ti','O']

# options for how to generate Q-points for calculation
Qpoints_option = 'mesh' #'mesh' # mesh, file, or path

# for 'Qpoints_option' == 'mesh' ; 
Q_mesh_H = [ 2.5, 3.5, 21]
Q_mesh_K = [ 0.5, 1.5, 21]
Q_mesh_L = [-0.5, 0.5, 30]

# number of processes to split Q-point parallelization over
num_Qpoint_procs = 16






