
"""

Notes:
    - i had some security issue on ubuntu 22.04 about shared memory permissions. a warning
    is printed when running with MPI saying something about degraded performance. see here: 
        https://github.com/microsoft/WSL/issues/3397. 

    i 'fixed' it by setting 
        kernel.yama.ptrace_scope = 0 
    in the file
        /etc/sysctl.d/10-ptrace.conf
    and then restarting. 

    i am sure this is a security risk somehow and should be done at your own risk! alternatively
    you can ignore the warning or not use MPI at all. or, set to 0, run, and then revert once done. 

"""

import h5py 
import numpy as np

# --------------------------------------------------------------------------------------------------

# check if h5py built w/ MPI support
have_mpi = h5py.get_config().mpi
if not have_mpi:
    print('\n*** WARNING ***\nh5py not mpi enabled!')
    rank = 0
    size = 1
else:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

if rank == 0 and have_mpi:
    print(f'\nusing {size} MPI ranks')

# --------------------------------------------------------------------------------------------------

class c_compressor:
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,lammps_traj_file,hdf5_traj_file,num_steps=None,header_length=9):
        
        """
        class to read lammps trajectory file and write it to hdf5 file. this file is designed 
        to work in parallel
        """
        
        self.header_length = header_length
        self.lammps_traj_file = lammps_traj_file
        self.hdf5_traj_file = hdf5_traj_file

        #self.positions = np.zeros(self.num_atoms,)
        #self.types = np.zeros(self.num_atoms)

        #if rank == 0:
        #    self._get_meta_data(num_steps)
        #    self._split_trajectory_over_ranks()

        self._get_meta_data(num_steps)
        self._broadcast_steps()

        # all ranks have to create datasets etc.
        self._init_hdf5_file()

    # ----------------------------------------------------------------------------------------------

    def _broadcast_steps(self):

        """
        """

        pass        

    # ----------------------------------------------------------------------------------------------

    def _init_hdf5_file(self):

        """
        create output hdf5 file and pos, type, box_vector arrays in it
        """

        _num_steps = self.num_steps
        _num_atoms = self.num_atoms

        print('a')
        if have_mpi:
            db = h5py.File(self.hdf5_traj_file,'w',driver='mpio',comm=comm)
        else:
            db = h5py.File(self.hdf5_traj_file,'w')

        db.create_dataset('cartesian_pos',shape=(_num_steps,_num_atoms,3),dtype=float)        
        db.create_dataset('types',shape=_num_atoms,dtype=float)
        db.create_dataset('box_vectors',shape=(_num_steps,3,3),dtype=float)
        
        print('b')
        db.close()

    # ----------------------------------------------------------------------------------------------

    def _split_trajectory_over_ranks(self):

        """
        split the time steps over the ranks, so each rank knows what to do
        """

        self.time_steps = np.arange(self.num_steps)
        self.steps_on_ranks = np.array_split(self.time_steps,size)
        self.num_steps_on_ranks = [len(_) for _ in self.steps_on_ranks]

        print('\nnum steps on ranks:')
        [print(f'rank[{rank}]:',num) for rank, num in enumerate(self.num_steps_on_ranks)]

    # ----------------------------------------------------------------------------------------------
    
    def _get_meta_data(self,num_steps=None):

        """
        get number of atoms, number of time steps, and number of lines in the file. if num steps 
        is already known, we can save a little time by not scanning the whole file to find the 
        number of lines
        """
        
        with open(self.lammps_traj_file,'r') as _f:
            for ii in range(3):
                _f.readline()
            self.num_atoms = int(_f.readline().strip())
            
        if num_steps is None:
            with open(self.lammps_traj_file, "rb") as _f:
                self.num_lines = sum(1 for _ in _f)
            self.num_steps = int(self.num_lines/(self.header_length+self.num_atoms))
        else:
            self.num_steps = int(num_steps)
            self.num_lines = (self.num_atoms+self.header_length)*self.num_steps
        
        print('\nfile name:',self.lammps_traj_file)
        print('num atoms:',self.num_atoms)
        print('num lines:',self.num_lines)
        print('num steps:',self.num_steps)
    
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

lammps_traj_file = 'pos.dat'
hdf5_traj_file = 'pos.hdf5'
compressor = c_compressor(lammps_traj_file,hdf5_traj_file,num_steps=2501) 














