
import numpy as np
import h5py
from scipy.interpolate import interpn

from diffit.m_code_utils import c_timer

# --------------------------------------------------------------------------------------------------

def get_exp_data(file_name,H,K,L):

    """
    get exp data on same grid as calculated data
    """

    with h5py.File(file_name,'r') as db:
        exp_intensity_full = db['signal'][...]
        exp_H = db['H'][...]; exp_K = db['K'][...]; exp_L = db['L'][...]

    shape = [H.size,K.size,L.size]

    H, K, L = np.meshgrid(H,K,L,indexing='ij')
    H = H.flatten(); K = K.flatten(); L = L.flatten()
    Q = np.c_[H,K,L]

    exp_intensity = interpn((exp_H,exp_K,exp_L),exp_intensity_full,Q,
            bounds_error=False,fill_value=None)

    exp_intensity.shape = shape

    with h5py.File('tmp.hdf5','w') as db:
        db.create_dataset('signal',data=exp_intensity)
    
    exit()

    return exp_intensity

# --------------------------------------------------------------------------------------------------

