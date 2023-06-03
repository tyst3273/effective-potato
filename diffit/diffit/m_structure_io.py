
import numpy as np
from diffit.m_code_utils import crash

      
# --------------------------------------------------------------------------------------------------

def write_xyz(file_name,pos,types,type_strings=None,append=False):

    """
    write an xyz file 
    """
    
    if type_strings is not None:
        types = types.astype(object)
        for ii, _s in enumerate(type_strings):
            _inds = np.flatnonzero(types == ii)
            types[_inds] = _s
    
    if append:
        mode = 'a'
    else:
        mode = 'w'

    with open(file_name,mode) as _f:
        _f.write(f'{pos.shape[0]:d}\n ---\n')
        for ii in range(pos.shape[0]):
            _f.write(f'{types[ii]:4} {pos[ii,0]: 14.9f} {pos[ii,1]: 14.9f} {pos[ii,2]: 14.9f}\n')

# --------------------------------------------------------------------------------------------------

def write_lammpstrj(file_name,pos,types,vecs,append=True,sort_by_type=False):
    
    """
    write lammpstrj file
    """

    lx = vecs[0,0]
    ly = vecs[1,1]
    lz = vecs[2,2]

    xmin = pos[:,0].min(); xmax = xmin+lx
    ymin = pos[:,1].min(); ymax = ymin+ly
    zmin = pos[:,2].min(); zmax = zmin+lz
    
    if append:
        mode = 'a'
    else:
        mode = 'w'

    if sort_by_type:
        inds = np.argsort(types)
        types = types[inds]
        pos = pos[inds,:]

    num_atoms = pos.shape[0]

    with open(file_name,mode) as _f:
        _f.write('ITEM: TIMESTEP\n0\n')
        _f.write(f'ITEM: NUMBER OF ATOMS\n{num_atoms}\n')
        _f.write('ITEM: BOX BOUNDS pp pp pp\n')
        _f.write(f'{xmin: .9f} {xmax:.9f}\n{ymin: .9f} {ymax:.9f}\n{zmin: .9f} {zmax:.9f}\n')
        _f.write('ITEM: ATOMS id type x y z\n')
        for ii in range(num_atoms):
            _f.write(f'{ii+1:<6g} {types[ii]+1:3g} {pos[ii,0]: 14.9f} {pos[ii,1]: 14.9f}' \
                                    f' {pos[ii,2]: 14.9f}\n')
        
# --------------------------------------------------------------------------------------------------

def write_poscar(file_name,crystal):

    """
    write a poscar file 
    """

    latvecs = crystal.sc_vectors
    type_str = crystal.basis_type_strings 
    type_nums = crystal.sc_type_nums
    unique_nums, type_counts = np.unique(type_nums,return_counts=True)
    num_types = type_str.size

    pos = crystal.sc_positions_reduced


    with open(file_name,'w') as _f:
        _f.write('written by diffit!\n')
        _f.write('1.0\n')

        msg = ''
        for ii in range(3):
            for jj in range(3):
                msg += f' {latvecs[ii,jj]: 12.6f}'
            msg += '\n'
        _f.write(msg)

        for ii in range(num_types):
            _f.write(f' {type_str[ii]}')
        _f.write('\n')
        for ii in range(num_types):
            _f.write(f' {type_counts[ii]}')
        _f.write('\nDirect\n')

        for ii in range(num_types):
            inds = np.flatnonzero(type_nums == ii)
            for jj in range(inds.size):
                ind = inds[jj]
                _f.write(f' {pos[ind,0]: 12.6f} {pos[ind,1]: 12.6f} {pos[ind,2]: 12.6f}\n')

            
# --------------------------------------------------------------------------------------------------

def write_psf_hdf5(file_name,pos,types):

    """
    write an hdf5 file that can be read by my 'pynamic-structure-factor' code
    """
    
    pass
            
# --------------------------------------------------------------------------------------------------

 







    






