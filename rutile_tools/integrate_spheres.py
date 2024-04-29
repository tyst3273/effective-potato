
#from m_integrate_rods_mod import c_integrate_rods, get_uvw
from m_integrate_rods import c_integrate_rods

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import SymLogNorm as norm
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import yaml
import time

viridis = mpl.colormaps['viridis']
viridis.set_bad('w')


def redirect(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull,'w')
        vals = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return vals
    return wrapper
            
# --------------------------------------------------------------------------------------------------

def get_rotations():
    
    with open('symm.yaml','r') as stream:
        symm = yaml.safe_load(stream)
    symm_ops = symm['space_group_operations']
    num_ops = len(symm_ops)
    rotations = np.zeros((num_ops,3,3,),dtype=float)
    for ii in range(num_ops):
        rot = symm_ops[ii]['rotation']
        rotations[ii,...] = rot
    return rotations
    
# --------------------------------------------------------------------------------------------------

#@redirect
def integrate_sphere(Q,file_name='293K_quenched.hdf5',radius=0.1,
                 path='/home/ty/research/projects/materials/flashed_rutile/reduced_data'):
    
    Q = np.array(Q,dtype=float)
    file_name = os.path.join(path,file_name)
    scale = 1e6
    
    # recip lat vecs
    b = np.array([[1.36265134, 0.        , 0.        ],
                  [0.        , 1.36265134, 0.        ],
                  [0.        , 0.        , 2.11057619]])
    Qc = b@Q # center of cut; need in cartesian coords
    
    # integrate the data
    u_binning = radius*2 # 1/A
    v_binning = radius*2 # 1/A
    w_binning = radius*2 # 1/A
    
    # load datasets and integrate data
    data = c_integrate_rods(file_name)
    #ata.integrate_sphere(Qc,radius)
    data.plot_sphere_volume(Qc,radius,stride=4)

    # get the integrated data
    sig = data.integrated_signal*scale
    err = data.integrated_error*scale
    
    return sig, err, Q

# --------------------------------------------------------------------------------------------------

#file_names = ['100C_000mA_no_symm.hdf5','100C_040mA_no_symm.hdf5','100C_300mA_no_symm.hdf5',
#              '100C_500mA_no_symm.hdf5','293K_annealed_no_symm.hdf5','293K_quenched_no_symm.hdf5']

file_names = ['100C_000mA_no_symm.hdf5'] #,'293K_quenched_no_symm.hdf5']

braggs = np.loadtxt('outlier_peaks_symm',dtype=int)
rotations = get_rotations()

num_braggs = braggs.size
num_rotations = rotations.shape[0]

with h5py.File('bragg_peaks.hdf5','a') as db:
        
    # loop over files
    for file_name in file_names:    
        
        # delete group if it exists
        if file_name in db.keys():
            del db[file_name]
        db.create_group(file_name)
        
        # loop over outlier peaks
        for ii in range(num_braggs):
                
            Q = braggs[ii,:]
            Q = [2,1,2]
                
            # loop over symmetry operations
            for jj in range(num_rotations):
                
                rot = rotations[jj,...]
                Qp = rot@Q
                    
                Q_label = f'Q=({Qp[0]:.0f},{Qp[1]:.0f},{Qp[2]:.0f})'
                _label = os.path.join(file_name,Q_label)
                    
                # if this Q is already done, skip
                if Q_label in db[file_name].keys():
                    continue
                    
                _g = db.create_group(_label)#file_name+'/'+Q_label)
                    
                # integrate the data
                print(file_name)
                print(Q_label)
                print('\n')
                sig, err, _Qp = integrate_sphere(Qp,file_name=file_name,radius=0.1)

                # write the data to hdf5 file
                _g.create_dataset('Q',data=Qp)
                _g.create_dataset('signal',data=sig)
                _g.create_dataset('error',data=err)
                    
