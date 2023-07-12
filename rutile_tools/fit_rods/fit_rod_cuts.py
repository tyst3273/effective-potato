
import h5py
import numpy as np
import matplotlib.pyplot as plt

a = 2*np.pi/4.611; c = 2*np.pi/2.977
rlat = np.array([[a,0,0],[0,a,0],[0,0,c]])
rinv = np.linalg.inv(rlat)

with h5py.File('293K_quenched_rod_cuts.hdf5','r') as db:
    
    projs = list(db.keys())
    projs.pop(projs.index('u_binning'))
    projs.pop(projs.index('v_binning'))
    projs.pop(projs.index('w_binning'))
    
    steps = list(db[projs[0]].keys())
    steps.pop(steps.index('uvw'))
    
    # loop over projections
    for which in projs:
        
        gp = db[which]
        uvw = gp['uvw'][...]
        u = uvw[:,0]; v = uvw[:,1]; w = uvw[:,2]
        
        for s in steps:
            
            u_bins = gp[s]['u_bins'][...]
            v_bins = gp[s]['v_bins'][...]
            w_bins = gp[s]['w_bins'][...]
            Q = gp[s]['Q_center'][...]
            signal = gp[s]['signal'][...]

            extent = [v_bins.min(),v_bins.max(),w_bins.min(),w_bins.max()]
            vmax = signal.max()*0.9
            fig, ax = plt.subplots(figsize=(8,8))
            im = ax.imshow(signal,cmap='viridis',interpolation='none',vmin=0,vmax=vmax,
                aspect='auto',origin='lower',extent=extent)
            fig.colorbar(im,ax=ax,extend='both')

            plt.show()
            plt.clf()
            plt.close()


            #center = fit_signal(signal,v_bins,w_bins)
        
