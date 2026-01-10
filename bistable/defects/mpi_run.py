
import numpy as np
import h5py
import sys
import os

from mpi4py import MPI
from m_bistable_defects import c_bistable_defects

# --------------------------------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()

# --------------------------------------------------------------------------------------------------

ny = 501
nv = 501
y = np.linspace(0.0,0.2,ny)
v = np.linspace(0.1,0.4,nv)
y_grid, v_grid = np.meshgrid(y,v,indexing='ij')
y_grid = y_grid.flatten()
v_grid = v_grid.flatten()
num = ny * nv

# --------------------------------------------------------------------------------------------------

inds = np.arange(num)
split = np.array_split(inds,num_ranks)
num_on_rank = [_.size for _ in split]
my_num = num_on_rank[rank]
my_inds = split[rank]

if rank == 0:

    msg = '\n*** PARAL ***'
    for ii in range(num_ranks):
        msg += f'\n proc[{ii:02d}]: {split[ii].size:6d}'
    print(msg+'\n')

    multistable = np.zeros(num,dtype=bool)
    n_diff = np.zeros(num,dtype=float)
    x_diff = np.zeros(num,dtype=float)
    n_lo = np.zeros(num,dtype=float)
    x_lo = np.zeros(num,dtype=float)
    n_hi = np.zeros(num,dtype=float)
    x_hi = np.zeros(num,dtype=float)

my_multistable = np.zeros(my_num,dtype=bool)
my_n_diff = np.zeros(my_num,dtype=float)
my_x_diff = np.zeros(my_num,dtype=float)
my_n_lo = np.zeros(my_num,dtype=float)
my_x_lo = np.zeros(my_num,dtype=float)
my_n_hi = np.zeros(my_num,dtype=float)
my_x_hi = np.zeros(my_num,dtype=float)

# --------------------------------------------------------------------------------------------------

count = 0
for ii, ind in enumerate(my_inds):
    
    yy = y_grid[ind]
    vv = v_grid[ind]

    # _stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

    bistable = c_bistable_defects(vv,yy)
    my_multistable[ii], my_n_lo[ii], my_x_lo[ii], my_n_hi[ii], my_x_hi[ii], \
        my_n_diff[ii], my_x_diff[ii] = bistable.solve()
    
    # sys.stdout.close()
    # sys.stdout = _stdout

    if rank == 0:

        print(f'count: {count}',flush=True)
        count += 1

# --------------------------------------------------------------------------------------------------

if rank == 0:

    multistable[my_inds] = my_multistable[...]
    n_lo[my_inds] = my_n_lo[...]
    x_lo[my_inds] = my_x_lo[...]
    n_hi[my_inds] = my_n_hi[...]
    x_hi[my_inds] = my_x_hi[...]
    n_diff[my_inds] = my_n_diff[...]
    x_diff[my_inds] = my_x_diff[...]

    for rr in range(1,num_ranks):

        _inds = split[rr]
        _num = _inds.size

        _multistable = np.zeros(_num,dtype=bool)
        comm.Recv(_multistable,source=rr,tag=1)
        multistable[_inds] = _multistable[...]
        
        _n_lo = np.zeros(_num,dtype=float)
        comm.Recv(_n_lo,source=rr,tag=2)
        n_lo[_inds] = _n_lo[...]

        _x_lo = np.zeros(_num,dtype=float)
        comm.Recv(_x_lo,source=rr,tag=3)
        x_lo[_inds] = _x_lo[...]

        _n_hi = np.zeros(_num,dtype=float)
        comm.Recv(_n_hi,source=rr,tag=4)
        n_hi[_inds] = _n_hi[...]

        _x_hi = np.zeros(_num,dtype=float)
        comm.Recv(_x_hi,source=rr,tag=5)
        x_hi[_inds] = _x_hi[...]

        _n_diff = np.zeros(_num,dtype=float)
        comm.Recv(_n_diff,source=rr,tag=6)
        n_diff[_inds] = _n_diff[...]

        _x_diff = np.zeros(_num,dtype=float)
        comm.Recv(_x_diff,source=rr,tag=7)
        x_diff[_inds] = _x_diff[...]

else:

    comm.Send(my_multistable,dest=0,tag=1)
    comm.Send(my_n_lo,dest=0,tag=2)
    comm.Send(my_x_lo,dest=0,tag=3)
    comm.Send(my_n_hi,dest=0,tag=4)
    comm.Send(my_x_hi,dest=0,tag=5)
    comm.Send(my_n_diff,dest=0,tag=6)
    comm.Send(my_x_diff,dest=0,tag=7)

# --------------------------------------------------------------------------------------------------

# write results to hdf5 file
if rank == 0:

    with h5py.File('results.h5','w') as db:

        db.create_dataset('multistable',data=multistable.reshape(ny,nv))
        db.create_dataset('n_lo',data=n_lo.reshape(ny,nv))
        db.create_dataset('x_lo',data=x_lo.reshape(ny,nv))
        db.create_dataset('n_hi',data=n_hi.reshape(ny,nv))
        db.create_dataset('x_hi',data=x_hi.reshape(ny,nv))
        db.create_dataset('n_diff',data=n_diff.reshape(ny,nv))
        db.create_dataset('x_diff',data=x_diff.reshape(ny,nv))
        db.create_dataset('y',data=y)
        db.create_dataset('v',data=v)
    
