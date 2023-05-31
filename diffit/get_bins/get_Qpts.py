
import h5py 
import numpy as np

Q_center = [3,1,0]

with h5py.File('k1_u_binning.hdf5','r') as db:
    
    num_bins = db['bin_centers'].size

    nQ = 0
    for ii in range(num_bins):
        nQ += db[f'weights_in_bin_{ii}'].size

    Q = np.zeros((nQ,3))
    
    shift = 0
    for ii in range(num_bins):

        Q_in_bin = db[f'Q_in_bin_{ii}'][...]
        n = Q_in_bin.shape[0]

        Q[shift:shift+n,:] = Q_in_bin[...]

        shift += n

np.savetxt(f'Q_H{Q_center[0]:3.2f}_K{Q_center[1]:3.2f}_L{Q_center[2]:3.2f}',Q,fmt='%9.6f')

    

