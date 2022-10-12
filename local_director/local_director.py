
import numpy as np
import h5py




class c_local_director:


    def __init__(self,sc_size=[16,16,16],temp=250):

        """
        lattice vectors are calculated from sc size and box vector
        """

        self.temp = temp
        self.sc_size = sc_size

        self.vertex_type = 2


    def _get_block_inds(self,num_steps,block_size):

        """
        set up indices to loop over to read a file
        """

        num_blocks = int(np.ceil(num_steps/block_size))
        
        block_inds = np.zeros((num_blocks,2),dtype=int)

        _blocks = np.arange(num_blocks)
        block_inds[:,0] = _blocks*block_size
        block_inds[:,1] = (_blocks+1)*block_size

        _remainder = num_steps % block_size
        block_inds[-1,1] = block_inds[-1,0]+_remainder
        
        return num_blocks, block_inds


    def _get_reduced_pos(self,lat_vecs,pos):

        """
        get positions of atoms in reduced coords
        """

        red = np.zeros(pos.shape)
        for ii in range(red.shape[0]): 
            inv_lat_vecs = np.linalg.inv(lat_vecs[ii,:,:])
            for jj in range(red.shape[1]):
                red[ii,jj,:] = pos[ii,jj,0]*inv_lat_vecs[0,:]+ \
                               pos[ii,jj,1]*inv_lat_vecs[1,:]+ \
                               pos[ii,jj,2]*inv_lat_vecs[2,:]

                print(ii,jj)

        return red


    def get_vertex_site_map(self,block_size=200):

        """
        map octahedral vertex atoms onto thier 'sites', i.e. their neighboring metal atoms
        """

        self.input_file = f'mapi_{self.temp:g}K.hdf5'
        self.output_file = f'vertex_map_{self.temp:g}K.hdf5'

        with h5py.File(self.input_file,'r') as _db:
        
            num_steps = _db['cartesian_pos'].shape[0]
            num_atoms = _db['types'].shape[0]
            types = _db['types'][:]

            num_blocks, block_inds = self._get_block_inds(num_steps,block_size)

            vertex_inds = np.flatnonzero(types)
            num_vertex = vertex_inds.size

            for bb in range(num_blocks):

                inds = block_inds[bb,:]

                vertex_pos = _db['cartesian_pos'][inds[0]:inds[1],:,:]
                vertex_pos = vertex_pos[:,vertex_inds,:]
                box_vectors = _db['box_vectors'][inds[0]:inds[1],:,:]

                vertex_pos = self._get_reduced_pos(box_vectors,vertex_pos)
                
                print(vertex_pos[0,:,:])
                exit()
            


# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    
    local_director = c_local_director()

    # need to figure out where stuff is
    local_director.get_vertex_site_map()











