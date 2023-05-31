
import numpy as np
import h5py
from scipy.interpolate import interpn

from diffit.m_code_utils import c_timer

# --------------------------------------------------------------------------------------------------

def get_inds_and_weights(file_name):

    with h5py.File(file_name,'r') as db:

        bin_centers = db['bin_centers'][...]
        num_bins = bin_centers.size

        inds = [[] for _ in range(num_bins)]
        weights = [[] for _ in range(num_bins)]

        shift = 0
        for ii in range(num_bins):
            w = db[f'weights_in_bin_{ii}'][...]
            inds[ii] = list(range(shift,shift+w.size))
            shift += w.size
            weights[ii] = w

    return inds, weights, bin_centers

# --------------------------------------------------------------------------------------------------

def put_into_bins(calc,inds,weights):
    
    num_bins = len(weights)
    calc_bins = np.zeros(num_bins)

    for ii in range(num_bins):
        calc_bins[ii] = np.sum(calc[inds[ii]]*weights[ii])

    return calc_bins

# --------------------------------------------------------------------------------------------------

