
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

def write_poscar(file_name,pos,types,type_strings=None):

    """
    write a poscar file 
    """
    
    pass
            
# --------------------------------------------------------------------------------------------------

def write_psf_hdf5(file_name,pos,types):

    """
    write an hdf5 file that can be read by my 'pynamic-structure-factor' code
    """
    
    pass
            
# --------------------------------------------------------------------------------------------------

 







    






