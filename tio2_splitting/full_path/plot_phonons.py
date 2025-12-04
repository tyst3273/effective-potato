
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

ha_2_meV = 27211.39613179

def get_dists(qpts):

    num_qpts = qpts.shape[0]

    dists = np.zeros(num_qpts)
    for ii in range(1,num_qpts):
        q1 = qpts[ii,:]
        q0 = qpts[ii-1,:]
        dists[ii] = dists[ii-1] + np.linalg.norm(q1-q0)
    
    dists /= dists.max()
    return dists

with nc.Dataset('run_PHBST.nc','r') as ds:

    bands = np.array(ds['phfreqs']) * ha_2_meV # meV
    angmom = np.array(ds['phangmom']) * ha_2_meV # meV * s
    num_qpts, num_modes = bands.shape

    dists = get_dists(np.array(ds['qpoints']))
    
    x = np.linspace(0,1,num_qpts)

for bb in range(num_modes):

    plt.plot(x,bands[:,bb],c='k',ls='-',lw=1)

    L_len = np.sqrt( np.sum(angmom[:,bb]**2, axis=1) ) / 500
    hi = bands[:,bb] + L_len
    lo = bands[:,bb] - L_len

    plt.fill_between(x,lo,hi,color='m',alpha=0.2)
    

plt.show()