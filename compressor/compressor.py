
"""
Notes:
    - i had some security issue on ubuntu 22.04 about shared memory permissions. see here: 
        https://github.com/microsoft/WSL/issues/3397. 
    i 'fixed' it by setting 
        kernel.yama.ptrace_scope = 0 
    in the file
        /etc/sysctl.d/10-ptrace.conf
    and then restarting. 
    i am sure this is a security risk somehow and should be done at your own risk! or don't
    use MPI. alternatively, set to 0, run, and then revert once done. 
"""

import h5py 
import numpy as np

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

if rank == 0:
    print(size)

# --------------------------------------------------------------------------------------------------

class c_compressor:
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,file_name,num_steps=None,header_length=9):
        
        """
        class to read lammps trajectory file and write it to hdf5 file. this file is designed 
        to work in parallel
        """
        
        self.header_length = header_length
        self.file_name = file_name

        #self.positions = np.zeros(self.num_atoms,)
        #self.types = np.zeros(self.num_atoms)
        
    # ----------------------------------------------------------------------------------------------
    
    def _get_meta_data(self,num_steps=None):

        """
        get number of atoms, number of time steps, and number of lines in the file. if num steps 
        is already known, we can save a little time by not scanning the whole file to find the 
        number of lines
        """
        
        with open(self.file_name,'r') as _f:
            for ii in range(3):
                _f.readline()
            self.num_atoms = int(_f.readline().strip())
            
        if num_steps is None:
            with open(self.file_name, "rb") as _f:
                self.num_lines = sum(1 for _ in _f)
            self.num_steps = int(self.num_lines/(self.header_length+self.num_atoms))
        else:
            self.num_steps = int(num_steps)
            self.num_lines = (self.num_atoms+self.header_length)*self.num_steps
        
        print('\nfile name:',self.file_name)
        print('num atoms:',self.num_atoms)
        print('num lines:',self.num_lines)
        print('num steps:',self.num_steps)
    
    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

print(rank)

file_name = 'pos.dat'
#compressor = c_compressor(file_name,num_steps=2501)














