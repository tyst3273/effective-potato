

"""
Author: Tyler C. Sterling
Email: ty.sterling@colorado.edu
Affil: University of Colorado Boulder, Raman Spectroscopy and Neutron Scattering Lab
Date: 04/26/2023
Description: 
    tools to extract essential data from *.nxs CORELLI files and write them to
    *.hdf5. calculates background from spherical averaging and sums data to a
    single BZ. and other stuff.     
"""

from timeit import default_timer
import numpy as np
import h5py
import os


# --------------------------------------------------------------------------------------------------

def crash(err_msg=None,exception=None):
    """
    stop execution in a safe way
    """
    msg = '\n*** error ***\n'
    if err_msg is not None:
        msg += err_msg+'\n'
    if exception is not None:
        msg += '\nException:\n'+str(exception)+'\n'
    print(msg)
    raise KeyboardInterrupt

# --------------------------------------------------------------------------------------------------

def check_file(file_name):
    """
    check if the specified file exists
    """
    if not os.path.exists(file_name):
        msg = f'the file:\n  \'{file_name}\' \nwas not found!'
        crash(msg)

# --------------------------------------------------------------------------------------------------


class c_timer:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,label,units='s'):
        """
        small tool for timing and printing timing info
        """

        self.label = label
        if units == 'm':
            self.units = 'm'
            self.scale = 1/60
        elif units == 'ms':
            self.units = 'ms'
            self.scale = 1000
        else:
            self.units = 's'
            self.scale = 1
        self.start_time = default_timer()

    # ----------------------------------------------------------------------------------------------

    def stop(self):
        """
        stop timer and print timing info
        """

        elapsed_time = default_timer()-self.start_time
        elapsed_time *= self.scale
        msg = f'timing:   {self.label} {elapsed_time:9.5f} [{self.units}]'
        print(msg)

# --------------------------------------------------------------------------------------------------



class c_CORELLI_tools:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        class to interact with elastic scattering data from corelli
        """
        #self.ws_name = 'workspace'

    # ----------------------------------------------------------------------------------------------

    def import_mantid(self):
        """
        try to import mantid and print exception if fails
        """
        try:
            import mantid.simpleapi as msi
        except Exception as ex:
            msg = 'couldnt import mantid!'
            crash(msg,ex)
        self.msi = msi

    # ----------------------------------------------------------------------------------------------

    def check_ws(self,ws_name,crash=True):
        """
        check if a workspace exists in memory on mantid backend
        if crash=True, it crashes if ws not loaded.
        if crash=False, it just returns (bool) wheter or not its loaded
        """
        loaded = self.msi.mtd.doesExist(ws_name)
        if not crash:
            return loaded
        if not loaded:
            msg = f'the workspace isnt loaded!\n'
            crash(msg)

    # ----------------------------------------------------------------------------------------------

    def get_ws(self,ws_name):
        """
        get object handle for workspace that exists in mantid backend
        """
        self.check_ws(ws_name)
        return self.msi.mtd[ws_name]

    # ----------------------------------------------------------------------------------------------

    def load_ws(self,file_name,ws_name,force_load=False):
        """
        check if the MD file is already loaded. if not, load it
        """
        load = not(self.check_ws(ws_name,crash=False))
        if force_load or load:
            msg = f'\nloading workspace from file \'{file_name}\'\n'
            print(msg)
            self.msi.Load(Filename=file_name,OutputWorkspace=ws_name)
        else:
            msg = f'\nworkspace is already loaded! continuing...\n'
            print(msg)

    # ----------------------------------------------------------------------------------------------

    def load_nxs(self,nxs_file):
        """
        load the MDE workspace
        """
        timer = c_timer('load_MDE',units='m')
    
        ws_name = os.path.basename(nxs_file)
        if not ws_name.endswith('.nxs'):
            msg = '\nmust be a *.nxs file!\n'
            crash(msg)

        self.ws_name = ws_name[:-4]
        print('\nworkspace name:\n',self.ws_name)

        self.load_ws(file_name=nxs_file,ws_name=self.ws_name)
        timer.stop()

    # ----------------------------------------------------------------------------------------------

    def get_lattice_vectors_from_params(self):
        """
        take lattice params (cell lenghts/angles) and return lattice vector array
        note, lengths should be in Angstrom, angles in radians

        note, lattice_vectors are ROW vectors
        """
        lattice_vectors = np.zeros((3,3))
        lattice_vectors[0,:] = [self.a,0,0]
        lattice_vectors[1,:] = [self.b*np.cos(self.gamma),self.b*np.sin(self.gamma),0]
        lattice_vectors[2,:] = [self.c*np.cos(self.beta),
                                    -self.c*np.sin(self.beta)*np.cos(self.alpha),self.c]
        self.lattice_vectors = lattice_vectors

    # ----------------------------------------------------------------------------------------------

    def get_reciprocal_lattice_vectors(self):
        """
        get reciprocal lattice vectors from lattice vectors. we need them to go from HKL to 1/A
        note: the UnitCell class of mantid seems to offer methods for this (e.g. qFromHKL()) but
        I cant figure out how to use them. I hate mantid.
        """
        lattice_vectors = self.lattice_vectors
        _2pi = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)*2*np.pi
        self.recip_lattice_vectors = np.linalg.solve(lattice_vectors,_2pi).T

    # ----------------------------------------------------------------------------------------------

    def get_lattice(self):
        """
        get the lattice vectors for the data in the MDE file
        """
        ws = self.get_ws(self.ws_name)

        _exp_info = ws.getExperimentInfo(0)
        _sample = _exp_info.sample()

        if not _sample.hasOrientedLattice():
            msg = f'couldnt find OrientedLattice object in workspace\n'
            crash(msg)

        _uc = _sample.getOrientedLattice()

        # lattice vectors lenghts in Angstrom
        a = _uc.a1(); b = _uc.a2(); c = _uc.a3()
        self.a = a; self.b = b; self.c = c

        # unitcell angles in Radians
        alpha = _uc.alpha1(); beta = _uc.alpha2(); gamma = _uc.alpha3()
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

        # get matrices of lattice and reciprocal lattice vectors
        self.get_lattice_vectors_from_params()
        self.get_reciprocal_lattice_vectors()

    # ----------------------------------------------------------------------------------------------

    def get_dim_array(self,dim):
        """
        convert mantid dimension object to numpy array containing bin centers
        """
        _min = dim.getMinimum()
        _max = dim.getMaximum()
        _nbins = dim.getNBins()
        _d = (_max-_min)/_nbins
        bins = np.arange(_min+_d/2,_max,_d)
        return bins

    # ----------------------------------------------------------------------------------------------

    def get_hdf5_from_nxs_file(self,nxs_file,hdf5_file,swap_yz=False):
        """
        strip H, K, L, signal, and error from the *.nxs file and write it to hdf5
        """
        timer = c_timer('get_hdf5_from_nxs')

        check_file(nxs_file)
        self.import_mantid()
        self.load_nxs(nxs_file)

        self.get_lattice()

        ws = self.get_ws(self.ws_name)

        signal = np.array(ws.getSignalArray())
        error = np.sqrt(np.array(ws.getErrorSquaredArray()))

        if swap_yz:

            # assumes Y = L and Z = K. for some reason, Feng binned them this way
            H = self.get_dim_array(ws.getXDimension())
            K = self.get_dim_array(ws.getZDimension())
            L = self.get_dim_array(ws.getYDimension())
            
            signal = np.swapaxes(signal,1,2)
            error = np.swapaxes(error,1,2)

        else:

            # assumes 1st, 2nd, 3rd dims actually are H, K, L respectively
            H = self.get_dim_array(ws.getXDimension())
            K = self.get_dim_array(ws.getYDimension())
            L = self.get_dim_array(ws.getZDimension())

        shape = [H.size,K.size,L.size]

        with h5py.File(hdf5_file,'w') as o_db:

            o_db.create_dataset('H',data=H)
            o_db.create_dataset('K',data=K)
            o_db.create_dataset('L',data=L)
            o_db.create_dataset('shape',data=shape)
            o_db.create_dataset('signal',data=signal)
            o_db.create_dataset('error',data=error)
            o_db.create_dataset('lattice_vectors',data=self.lattice_vectors)
            o_db.create_dataset('reciprocal_lattice_vectors',data=self.recip_lattice_vectors)

        timer.stop()

    # ----------------------------------------------------------------------------------------------

    def get_cartesian_coords(self,H,K,L):
        """
        go from rlu to cartesian coords
        """
        timer = c_timer('get_cartesian_coords',units='s')

        b = self.recip_lattice_vectors

        Qx = b[0,0]*H+b[1,0]*K+b[2,0]*L
        Qy = b[0,1]*H+b[1,1]*K+b[2,1]*L
        Qz = b[0,2]*H+b[1,2]*K+b[2,2]*L

        timer.stop()

        return Qx, Qy, Qz

    # ----------------------------------------------------------------------------------------------

    def get_spherical_coords(self,Qx,Qy,Qz):
        """
        get spherical coords from cartesian coords
        """

        timer = c_timer('get_spherical_coords',units='s')

        # get Q in spherical coords
        Q = np.sqrt(Qx**2+Qy**2+Qz**2)

        mask = (Q < 1e-6).astype(int)
        Q += mask # shift all 0's to 1.0

        polar = np.arccos(Qz/Q)
        azi = np.arctan2(Qy,Qx)

        Q += -mask # shift back to 0's

        timer.stop()

        return Q, polar, azi

    # ----------------------------------------------------------------------------------------------

    def _copy_header_datasets(self,source_db,dest_db):
        """
        copy 'header' info from database_1 to database_2
        """
        timer = c_timer('copy_datasets')

        print('\ncopying_datasets ...\n')

        keys = list(source_db.keys())

        # don't want to copy raw signal. will write new data later 
        keys.pop(keys.index('signal'))
        keys.pop(keys.index('error'))

        for key in keys:
            dest_db.create_dataset(key,data=source_db[key])

        timer.stop()

    # ----------------------------------------------------------------------------------------------

    def get_background_from_spherical_avg(self,hdf5_file,bg_file,delta_Q=0.1,punch_radius=0.25,
                        plot_histogram=False,fwhm=None):
        """
        punch out the bragg peaks and determine background at each |Q| as spherical average 
        of the data at same |Q|. 

        delta_Q is the step size for the spherical avg. bins
        punch_radius is the radius around each Bragg peak to punch out.
        """

        timer = c_timer('get_background_from_spherical_avg')

        # get number of Q-points and reciprocal lattice vectors in file
        with h5py.File(hdf5_file,'r') as i_db, h5py.File(bg_file,'w') as o_db:

            self.recip_lattice_vectors = i_db['reciprocal_lattice_vectors'][...]
            self._copy_header_datasets(i_db,o_db)

            # get Q in rlu
            H = i_db['H'][...]
            K = i_db['K'][...]
            L = i_db['L'][...]
            signal = i_db['signal'][...].flatten()

        self.shape = [H.size,K.size,L.size]
        
        H, K, L = np.meshgrid(H,K,L,indexing='ij')
        H = H.flatten(); K = K.flatten(); L = L.flatten()
        self.num_Q = H.size

        inds_to_keep = np.flatnonzero(~np.isnan(signal))
        inds_to_keep = np.intersect1d(inds_to_keep,np.flatnonzero(~np.isinf(signal)))

        # get Q in carestian coords
        Qx, Qy, Qz = self.get_cartesian_coords(H,K,L)

        # get the distance from each Q-pt to the nearest Bragg peak in 1/A
        dist_to_bragg = self._get_distance_to_braggs(Qx,Qy,Qz,H,K,L)

        # indices of Q-pts that are far enough from Bragg peaks
        inds_to_keep = np.intersect1d(inds_to_keep,np.flatnonzero(dist_to_bragg >= punch_radius))

        # spherical polar coords
        Q, polar, azi = self.get_spherical_coords(Qx,Qy,Qz)

        # free up some memory
        del H, K, L, Qx, Qy, Qz

        # Q-pt bins
        Q_min = Q.min(); Q_max = Q.max()
        num_Q_bins = round((Q_max-Q_min)/delta_Q)

        # spherically avg'd signal by histogramming data far from bragg peaks
        sig_hist, Q_bins = self._spherical_average(Q[inds_to_keep],
                                signal[inds_to_keep],num_Q_bins)

        # optionally broaden the BG a little
        if not fwhm is None:
            sig_hist = self._gaussian_smooth(Q_bins,sig_hist,fwhm)

        # optionally plot the histogram
        if plot_histogram:
            import matplotlib.pyplot as plt
            plt.plot(Q_bins,sig_hist)
            plt.ylabel('Intensity [arb. units]')
            plt.xlabel('|Q| [1/A]')
            plt.show()

        # put background onto original Q-pt mesh
        inds = np.digitize(Q,Q_bins[:-1],right=False)
        background = sig_hist[inds]
        background.shape = self.shape

        # write background to the file
        with h5py.File(bg_file,'a') as o_db:
            o_db.create_dataset('background',data=background)

        timer.stop()

    # ----------------------------------------------------------------------------------------------

    def _spherical_average(self,Q,signal,num_Q_bins):
        """
        spherical average the signal (ignoring Q-pts near bragg peaks) 
        """

        timer = c_timer('spherical_avg')

        # histogram the signal
        sig_hist, Q_bins = np.histogram(Q,bins=num_Q_bins,weights=signal)

        # count the number of Q-pts in each bin
        sig_counts, _ = np.histogram(Q,bins=num_Q_bins)

        # avg. the signal
        sig_hist = sig_hist/sig_counts

        # bin centers
        Q_bins = (Q_bins[1:]+Q_bins[:-1])/2

        timer.stop()

        return sig_hist, Q_bins

    # ----------------------------------------------------------------------------------------------

    def _get_distance_to_braggs(self,Qx,Qy,Qz,H,K,L):
        """
        get distance of each Q-pt to nearest Bragg peak in units of 1/A
        """

        timer = c_timer('distance_to_braggs')

        # want the miller index of each Q-pt, i.e. nearest Bragg peak
        H_int = np.rint(H).astype(np.int32)
        K_int = np.rint(K).astype(np.int32)
        L_int = np.rint(L).astype(np.int32)

        # get nearest Bragg peak in cartesian coords
        Qx_int, Qy_int, Qz_int = self.get_cartesian_coords(H_int,K_int,L_int)

        timer.stop()

        # distance to nearest Bragg peak
        return np.sqrt((Qx_int-Qx)**2+(Qy_int-Qy)**2+(Qz_int-Qz)**2)

    # ----------------------------------------------------------------------------------------------

    def _gaussian_smooth(self,x,y,fwhm):
        """
        smooth data using gaussian convolution. calls another method to actually do the smoothing
        """

        # gaussian stddev
        sigma = fwhm/np.sqrt(8*np.log(2))

        # step size along x axis
        dx = x[1]-x[0]

        # x axis for gaussian function
        gx = np.arange(-6*sigma,6*sigma,dx)
        gaussian = np.exp(-0.5*(gx/sigma)**2)

        smooth = self._gaussian_smooth_1d(x,y,gaussian)

        return smooth

    # ----------------------------------------------------------------------------------------------

    def _gaussian_smooth_1d(self,x,y,gaussian):
        """
        smooth data using gaussian convolution. if there are nans in input array, interpolate to
        remove the nans.
        """

        from scipy.signal import convolve

        # mask infinites with nans
        y = np.nan_to_num(y,nan=np.nan,posinf=np.nan,neginf=np.nan)

        nans = np.isnan(y)
        mask_nans = np.any(nans)

        if mask_nans:
            y_interpolated = np.interp(x,x[~nans],y[~nans])
        else:
            y_interpolated = np.copy(y)

        # handle 1d or 2d arrays separately
        smooth = convolve(y_interpolated, gaussian, mode='same', method='auto')

        # normalize to same norm as input data
        norm_y = np.sqrt(np.sum(y_interpolated**2))
        norm_smooth = np.sqrt(np.sum(smooth**2))

        if norm_smooth < 1e-6:
            smooth[:] = 0.0
        else:
            smooth *= norm_y/norm_smooth

        # replace indices with nans with nans
        smooth[nans] = np.nan

        return smooth

    # ----------------------------------------------------------------------------------------------

    def subtract_background(self,hdf5_file,bg_file):
        """
        subtract bacgkround in BG file from data file and write to BG_SUBTRACTED file
        """

        with h5py.File(hdf5_file,'a') as data_db, h5py.File(bg_file,'r') as bg_db:

            if 'bg_subtracted_signal' in data_db.keys():
                print('\nbg_subtracted_signal already exists. deleting it ...\n')
                del data_db['bg_subtracted_signal']
        
            data_db.create_dataset('bg_subtracted_signal',
                    data=data_db['signal'][...]-bg_db['background'][...])

    # ----------------------------------------------------------------------------------------------

    def sum_to_single_brillouin_zone(self,hdf5_file,summed_file,
                bragg_file='bragg_list.txt',subtracted=True):
        """
        sum all of the cuts in each extended BZ in list 'bragg_file' into single BZ. 
        use bg subtracted data from file if subtracted = True
        """

        with h5py.File(hdf5_file,'r') as i_db:

            # get Q in rlu
            H = i_db['H'][...]
            K = i_db['K'][...]
            L = i_db['L'][...]

        dh = np.round(H[1]-H[0],6); dk = np.round(K[1]-K[0],6); dl = np.round(L[1]-L[0],6)
        h = np.arange(-0.5,0.5+dh,dh)
        k = np.arange(-0.5,0.5+dk,dk)
        l = np.arange(-0.5,0.5+dl,dl)

        braggs = np.loadtxt(bragg_file)
        num_braggs = braggs.shape[0]

        summed_signal = np.zeros((h.size,k.size,l.size),dtype=float)

        H = np.round(H,6); K = np.round(K,6); L = np.round(L,6)

        count = 0
        for bb in range(num_braggs):

            # the BZ to do
            bragg = braggs[bb,:]

            # get inds in file
            H_inds, K_inds, L_inds = self._get_inds_to_sum(bragg,H,K,L)

            # check that full BZ is covered
            if H_inds.size != h.size or K_inds.size != k.size or L_inds.size != l.size:
                print('not a full BZ! continuing')
                continue
            count += 1
            print(count)

            # get the data
            summed_signal += self._get_summed_signal(hdf5_file,H_inds,K_inds,L_inds,subtracted)

        summed_signal /= count

        with h5py.File(summed_file,'w') as o_db:
            o_db.create_dataset('h',data=h)
            o_db.create_dataset('k',data=k)             
            o_db.create_dataset('l',data=l)
            o_db.create_dataset('summed_signal',data=summed_signal)

    # ----------------------------------------------------------------------------------------------

    def _get_summed_signal(self,hdf5_file,H_inds,K_inds,L_inds,subtracted):
        """ 
        get indices of all Qpts in the specified BZ
        """

        with h5py.File(hdf5_file,'r') as i_db:
            if subtracted:
                sig = i_db['bg_subtracted_signal'][H_inds,...]
            else:
                sig = i_db['signal'][H_inds,...]
            sig = sig[:,K_inds,:]
            sig = sig[...,L_inds]
            return sig

    # ----------------------------------------------------------------------------------------------

    def _get_inds_to_sum(self,bragg,H,K,L):
        """
        get indices of all Qpts in the specified BZ
        """

        _eps = 1e-6

        H_inds = np.intersect1d(np.flatnonzero(H >= bragg[0]-(0.5+_eps)),   
                                np.flatnonzero(H <= bragg[0]+(0.5+_eps)))
        K_inds = np.intersect1d(np.flatnonzero(K >= bragg[1]-(0.5+_eps)),
                                np.flatnonzero(K <= bragg[1]+(0.5+_eps)))
        L_inds = np.intersect1d(np.flatnonzero(L >= bragg[2]-(0.5+_eps)),
                                np.flatnonzero(L <= bragg[2]+(0.5+_eps)))

        return H_inds, K_inds, L_inds

    # ----------------------------------------------------------------------------------------------
    
    def get_angular_correction(self,hdf5_file,bg_file,punch_radius=0.3):
        """
        get angular correction to background in file
        """

        timer = c_timer('get_angular_correction')

        # get number of Q-points and reciprocal lattice vectors in file
        with h5py.File(hdf5_file,'r') as sig_db, h5py.File(bg_file,'r') as bg_db:

            self.recip_lattice_vectors = sig_db['reciprocal_lattice_vectors'][...]

            """
            # get Q in rlu
            H = sig_db['H'][...]
            K = sig_db['K'][...]
            L = sig_db['L'][...]

            signal = sig_db['signal'][...].flatten()
            background = bg_db['background'][...]
            """

            # get Q in rlu
            H = sig_db['H'][::2]
            K = sig_db['K'][::2]
            L = sig_db['L'][::2]

            signal = sig_db['signal'][::2,::2,::2].flatten()
            background = bg_db['background'][::2,::2,::2]

        signal = np.nan_to_num(signal,nan=0.0,posinf=0.0,neginf=0.0)

        self.shape = [H.size,K.size,L.size]
        print(self.shape)

        H, K, L = np.meshgrid(H,K,L,indexing='ij')
        H = H.flatten(); K = K.flatten(); L = L.flatten()
        self.num_Q = H.size

        # get Q in carestian coords
        Qx, Qy, Qz = self.get_cartesian_coords(H,K,L)

        # get the distance from each Q-pt to the nearest Bragg peak in 1/A
        dist_to_bragg = self._get_distance_to_braggs(Qx,Qy,Qz,H,K,L)

        # indices of Q-pts that are far enough from Bragg peaks
        keep = (dist_to_bragg >= punch_radius).astype(int)

        # spherical polar coords
        Q, polar, azi = self.get_spherical_coords(Qx,Qy,Qz)

        # free up some memory
        del H, K, L, Qx, Qy, Qz, Q

        signal.shape = self.shape
        keep.shape = self.shape

        import matplotlib.pyplot as plt
        #plt.imshow(((signal-background)*keep)[250,:,:].T,aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=5e-5)
        #plt.imshow(((signal-background)*keep)[:,100,:].T,aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=5e-5)
        plt.imshow(((signal-background)*keep)[:,:,163].T,aspect='auto',origin='lower',cmap='magma',vmin=0,vmax=5e-5)
        plt.show()

        timer.stop()

    # ----------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    corelli_tools = c_CORELLI_tools()

    hdf5_file = 'reduced_files/100C_500mA.hdf5'
    bg_file = 'reduced_files/100C_500mA_BG.hdf5'

    #corelli_tools.get_hdf5_from_nxs_file(nxs_file,hdf5_file)
    #corelli_tools.msi.DeleteWorkspace(corelli_tools.ws_name) # clean up memory

    #corelli_tools.get_background_from_spherical_avg(hdf5_file,bg_file=bg_file,
    #        delta_Q=0.02,punch_radius=0.3,plot_histogram=False,fwhm=None)

    #corelli_tools.subtract_background(hdf5_file,bg_file)

    #corelli_tools.sum_to_single_brillouin_zone(hdf5_file,summed_file,
    #        bragg_file='bragg_list.txt',subtracted=True)       















