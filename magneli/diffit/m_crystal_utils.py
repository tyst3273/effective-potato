
import numpy as np
from diffit.m_code_utils import crash, c_timer

      

# --------------------------------------------------------------------------------------------------

def change_coordinate_basis(vecs,coords):

    """
    go from one coordinate basis to the other using matrix of vectors
    """
    
    _1d = False
    if coords.ndim == 1:
        _1d= True
        coords = coords.reshape(1,3)
        
    _n = coords.shape[0]
    new_coords = np.zeros(coords.shape)
    for ii in range(_n):
        new_coords[ii,:] = vecs[0,:]*coords[ii,0]+vecs[1,:]*coords[ii,1]+vecs[2,:]*coords[ii,2]
        
    if _1d:
        new_coords = new_coords.reshape(3)
        
    return new_coords

# --------------------------------------------------------------------------------------------------

def do_minimum_image(origin,reduced_pos):

    """
    get relative position vectors between all positions in 'reduced_pos' array and 'origin'. 
    note that this is done in reduced coordinates since it's way easier and also is the only
    way to do it for non-orthorhombic cells
    """
    
    rel_pos = np.copy(reduced_pos)
    rel_pos[:,0] += -origin[0]; rel_pos[:,1] += -origin[1]; rel_pos[:,2] += -origin[2]
    
    _shift = -(rel_pos > 1/2).astype(int)
    _shift += (rel_pos <= -1/2).astype(int)
    
    rel_pos = rel_pos+_shift
    
    return rel_pos
            
# --------------------------------------------------------------------------------------------------

def get_neighbors(atom_pos,reduced_pos,lattice_vectors):

    """
    return neighbor vectors and distances between atom at atom_ind and all others.

    NOTE: the 0th element thats returned is always atom_ind
    """

    reduced_neighbor_vectors = do_minimum_image(atom_pos,reduced_pos)
    cart_neighbor_vectors = change_coordinate_basis(lattice_vectors,reduced_neighbor_vectors)

    neighbor_dist = np.sqrt(np.sum(cart_neighbor_vectors**2,axis=1))
    neighbors = np.argsort(neighbor_dist)
    neighbor_dist = neighbor_dist[neighbors]
    cart_neighbor_vectors = cart_neighbor_vectors[neighbors,:]

    return neighbors, neighbor_dist, cart_neighbor_vectors

# --------------------------------------------------------------------------------------------------
 
def get_neighbors_for_all_atoms_no_minimum_image(crystal,max_neighbors=10):

    """
    return neighbor vectors and distances between all atoms ignoring minimum image convention!
    i.e. this only works if the atoms are NOT wrapped.
    """

    num_atoms = crystal.num_sc_atoms
    neighbor_lists = np.zeros((num_atoms,max_neighbors),dtype=int)
    neighbor_dists = np.zeros((num_atoms,max_neighbors),dtype=float)

    pos = crystal.sc_positions_cart
    
    print(f'\ngetting neighbor distances... there are {num_atoms} atoms!\n')
    
    for ii in range(num_atoms):
        
        if ii % 1000 == 0:
            print(f'now on atom {ii}')

        _p = np.tile(pos[ii,:].reshape(1,3),reps=(num_atoms,1))
        _v = pos-_p
        _d = np.sqrt(np.sum(_v**2,axis=1))

        _sort = np.argsort(_d)

        neighbor_lists[ii,:] = _sort[:max_neighbors]
        neighbor_dists[ii,:] = _d[_sort][:max_neighbors]

    return neighbor_lists, neighbor_dists

# --------------------------------------------------------------------------------------------------







    






