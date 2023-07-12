
from m_integrate_custom import c_integrate_rods, get_uvw

import numpy as np
import matplotlib.pyplot as plt

no_symm = False
which = 'k1'
uvw = get_uvw(which) # unrotated matrix

summed_file = '../../summed_files/293K_quenched_summed.hdf5'

# load datasets and setup binning in rotated coords
summed = c_integrate_rods(summed_file,uvw) 

# get some data from the class
b = summed.recip_lat_vecs 
R = summed.rotation_matrix
u = summed.uvw[:,0]
v = summed.uvw[:,1]
w = summed.uvw[:,2]

Q = np.array([0,0,0],dtype=float)
Qc = b@Q # center of cut; need in cartesian coords
Qp = b@Q # center of plot; need in cartesian coords

# can do steps along u to cut perpendicular to u
Qc += u*0.3 # step along u by 0.2 A

# binning args
u_binning = 0.075 # 1/A 
v_binning = [-0.3,0.01,0.3] # 1/A
w_binning = [-0.3,0.01,0.3] # 1/A

# volumetric plot of the integration region (requires mayavi)
summed.plot_volume(Qc,u_binning,v_binning,w_binning,Q_plot_center=Qp)
    
# integrate the data 
summed.integrate(Qc,u_binning,v_binning,w_binning,num_procs=4)

# plot the cut
q_sig = summed.integrated_signal

x = summed.v_bin_centers
y = summed.w_bin_centers
nx = x.size; ny = y.size

q_sig.shape = [nx,ny]

fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(q_sig,cmap='Greys',interpolation='none',vmin=0,vmax=None,
    aspect='auto',origin='lower')
fig.colorbar(im,ax=ax,extend='both')

plt.show()



