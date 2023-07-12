
from m_integrate_custom import c_integrate_rods, get_uvw

import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_all_cuts(summed_file,out_file):
    
    projs = ['k1','k2','k3','k4','h1','h2','h3','h4']
    
    Q = np.array([0,0,0],dtype=float) # summed file is centered on 0,0,0
    
    # binning args
    u_binning = 0.1 # 1/A
    v_binning = [-0.4,0.01,0.4] # 1/A
    w_binning = [-0.4,0.01,0.4] # 1/A
    
    steps = np.arange(-0.6,0.7,0.1)
    
    with h5py.File(out_file,'w') as db:
        
        db.create_dataset('u_binning',data=u_binning)
        db.create_dataset('v_binning',data=v_binning)
        db.create_dataset('w_binning',data=w_binning)
    
        # loop over projections
        for which in projs:
    
            uvw = get_uvw(which) # unrotated matrix
        
            # load datasets and setup binning in rotated coords
            summed = c_integrate_rods(summed_file,uvw)
        
            # get some data from the class
            b = summed.recip_lat_vecs
            R = summed.rotation_matrix
            u = summed.uvw[:,0]
            v = summed.uvw[:,1]
            w = summed.uvw[:,2]
        
            Qp = b@Q # center of plot; need in cartesian coords
            
            db.create_group(which)
            db[which].create_dataset('uvw',data=uvw) # column vectors
            db[which]['uvw'].attrs['column_vecs'] = True
    
            # loop over steps along rod
            for ii, s in enumerate(steps):
        
                Qc = b@Q+u*s # center of cut; need in cartesian coords
        
                # volumetric plot of the integration region (requires mayavi)
                # summed.plot_volume(Qc,u_binning,v_binning,w_binning,Q_plot_center=Qp)
            
                # integrate the data 
                summed.integrate(Qc,u_binning,v_binning,w_binning,num_procs=4)
        
                # # plot the cut
                signal = summed.integrated_signal
        
                u_bins = summed.u_bin_centers
                v_bins = summed.v_bin_centers
                w_bins = summed.w_bin_centers
                
                signal = np.squeeze(signal.reshape(summed.grid_shape))
                
                db[which].create_group(f'step_{ii:d}')
                g = db[which][f'step_{ii:d}']
                g.create_dataset('Q_center',data=Qc)
                g.create_dataset('u_bins',data=u_bins)
                g.create_dataset('v_bins',data=v_bins)
                g.create_dataset('w_bins',data=w_bins)
                g.create_dataset('signal',data=signal)

                """
                extent = [v.min(),v.max(),w.min(),w.max()]
                vmax = signal.max()*0.9
                fig, ax = plt.subplots(figsize=(8,8))
                im = ax.imshow(signal,cmap='Greys',interpolation='none',vmin=0,vmax=vmax,
                    aspect='auto',origin='lower',extent=extent)
                fig.colorbar(im,ax=ax,extend='both')
            
                plt.show()
                plt.clf()
                plt.close()
                """

            break

if __name__ == '__main__':
    
    summed_file = '../../summed_files/293K_quenched_summed.hdf5'
    out_file = '293K_quenched_rod_cuts.hdf5'
    get_all_cuts(summed_file,out_file)
