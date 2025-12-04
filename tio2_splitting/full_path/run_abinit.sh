#!/bin/bash
#SBATCH --account=ucb542_asc2
#SBATCH --nodes=2
#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --error=err.out
#SBATCH --output=log.out
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --constraint=ib

module purge
module load intel/2022.1.2 impi/2021.5.0 mkl/2022.0.2 szip/2.1.1 netcdf/4.8.1 python/3.10.2 hdf5/1.12.1 pnetcdf/1.12.2 libxc/5.2.2

mpirun -np $SLURM_NTASKS /projects/tyst3273/software/abinit-10.4.7/build/src/98_main/abinit *.inp > log 2> err

