
import os
import numpy as np
import shutil


class c_lammps:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,dirs):

        """
        go into directories and run lammps calculations. its expected that the directories 
        contain a positions file with the name 'lammps.pos'
        """

        for d in dirs:
            if not os.path.exists(d):
                exit(f'\ncant calc in a non-existent dir!\n  \'{d}\'\n')

        self.dirs = dirs
        self.cwd = os.getcwd()

    # ----------------------------------------------------------------------------------------------

    def setup_jobs(self,lammps_in='in.lammps',force_field=None):

        """
        copy lammps input script and opitionally force_field file to all dirs
        """

        if not os.path.exists(lammps_in):
            exit('\nlammps input file must exists!\n')
        if force_field is not None: 
            if not os.path.exists(force_field):
               exit('\nforce field file mist exist if given!\n')

        for d in self.dirs:
            shutil.copy(lammps_in,d)

        if force_field is not None:
            for d in self.dirs:
                shutil.copy(force_field,d)

        self.lammps_in = lammps_in

    # ----------------------------------------------------------------------------------------------

    def run_lammps(self):
        
        """
        go and run the lammps calculations
        """

        for d in self.dirs:

            print('\ndir:',d)
            os.chdir(d)

            print(self.exe)
            os.system(self.exe)

            os.chdir(self.cwd)

    # ----------------------------------------------------------------------------------------------

    def setup_lammps(self,path_to_lammps='/usr/bin/lmp',path_to_mpi='/usr/bin/mpirun',
           num_mpi_procs=16,log_file=None):
        
        """
        set up the lammps execution
        """

        _i = os.path.split(self.lammps_in)[-1]

        self.exe = f'{path_to_mpi} -np {num_mpi_procs} {path_to_lammps} -i {_i}'
        
        if log_file is not None:
            self.exe += f' > {log_file}'

        print('\nlammps exe command:\n',self.exe,'\n')

    # ----------------------------------------------------------------------------------------------
