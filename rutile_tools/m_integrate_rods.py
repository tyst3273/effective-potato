
from timeit import default_timer
import numpy as np
import h5py
import os
import multiprocessing as mp


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


class c_integrate_rods:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,file_name,u=None,v=None,w=None,which=None,subtracted=False):
        """
        rotate coords to be aligned with rod and calculate intersection of bins in aligned
        coords with hkl coords

        NOTE: it is expected that u, v, w are orthonormal
        """
       
        check_file(file_name)
        self.file_name = file_name

        # whether or not to cut BG subtracted data
        self.subtracted = subtracted 

        # RECIPROCAL lattice constants
        self.a = 2*np.pi/4.611; self.c = 2*np.pi/2.977
        self.recip_lat_vecs = np.array([[self.a,0,0],
                                        [0,self.a,0],
                                        [0,0,self.c]])

        # which rod to integrate. optional, can give u,v,w manually.
        if which is not None:
            u, v, w = get_uvw(which)

        # uvw for rod binning -- NOTE: these are already in (1/A)
        self.uvw = np.array([u[:],v[:],w[:]]).T # column vecs
        u = self.uvw[:,0]; v = self.uvw[:,1]; w = self.uvw[:,2]

        # check that length == 1
        msg = 'u, v, and w should be orthonormal'
        if np.sqrt(np.sum(u**2))-1 > 0.01 or np.sqrt(np.sum(v**2))-1 > 0.01 \
        or np.sqrt(np.sum(w**2))-1 > 0.01:
            crash(msg)
        # check they are perpendicular
        if np.abs(u@v) > 0.01 or np.abs(u@w) > 0.01 or np.abs(v@w) > 0.01:
            crash(msg)

        # uvw for HKL binning. NOTE: these are all |h|=|k|=|l|=1 in units (1/A). 
        # our uvw are the same and all we want is to rotate to their frame. 
        self.hkl = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float).T # column vecs.

        self.rotation_matrix = np.linalg.solve(self.hkl,self.uvw).T

    # ----------------------------------------------------------------------------------------------

    def _get_rotated_cartesian_coords(self):
        """
        rotate to UVW aligned coords
        """

        R = self.rotation_matrix

        # cartesian coords in rotated frame
        self.Qxp = R[0,0]*self.Qx+R[0,1]*self.Qy+R[0,2]*self.Qz
        self.Qyp = R[1,0]*self.Qx+R[1,1]*self.Qy+R[1,2]*self.Qz
        self.Qzp = R[2,0]*self.Qx+R[2,1]*self.Qy+R[2,2]*self.Qz

    # ----------------------------------------------------------------------------------------------

    def _read_data_from_hdf5_file(self,Q,delta):
        """
        get coords from file and convert to cartesian (1/A)
        """

        # down sampling the data for speed right now
        with h5py.File(self.file_name,'r') as db:

            Qx = np.round(db['H'][...],4)*self.a
            Qy = np.round(db['K'][...],4)*self.a
            Qz = np.round(db['L'][...],4)*self.c

            Qx_inds = self._get_inds(Qx,Q[0],delta)
            Qy_inds = self._get_inds(Qy,Q[1],delta)
            Qz_inds = self._get_inds(Qz,Q[2],delta)

            Qx = Qx[Qx_inds]; Qy = Qy[Qy_inds]; Qz = Qz[Qz_inds]

            if self.subtracted:
                signal = db['bg_subtracted_signal'][Qx_inds,...]
            else:
                signal = db['signal'][Qx_inds,...]
            error = db['error'][Qx_inds,...]

        signal = signal[:,Qy_inds,:]
        self.signal = signal[...,Qz_inds]

        error = error[:,Qy_inds,:]
        self.error = error[...,Qz_inds]

        self.shape = [Qx.size,Qy.size,Qz.size]

        self.Qx, self.Qy, self.Qz = np.meshgrid(Qx,Qy,Qz,indexing='ij')

    # ----------------------------------------------------------------------------------------------

    def _get_inds(self,Q_arr,Q,delta):
        return np.intersect1d(np.flatnonzero(Q_arr >= Q-delta),
                              np.flatnonzero(Q_arr <= Q+delta))

    # ----------------------------------------------------------------------------------------------

    def butterworth_3d(self,x,y,z,wx,wy,wz,cx=0,cy=0,cz=0,n=40):
        """
        3d butterworth function; good approximation to a square pulse for large n but to 
        get fractional contribution from voxels near the edge, we can use lower n to 'smear' 
        the square shape a little
        """
        sx = wx/2; sy = wy/2; sz = wz/2 
        f = (1/(1+((x-cx)/sx)**n))*(1/(1+((y-cy)/sy)**n))*(1/(1+((z-cz)/sz)**n))
        return f

    # ----------------------------------------------------------------------------------------------

    def integrate(self,Q_center,u_binning,v_binning,w_binning,num_procs=1):
        """
        integrate around a Q-pt. Q is in cartesian coords, *_binning are the binning args
        along each UVW axis (i.e. rod binning). 
        if *_binning = [lo, d, hi], the binning is done from Q-lo to Q+hi with bin size d.
        if *_binning = d, the binning is done centered on Q with bin size d

        explanation: bin center coords are calculated as steps along u,v,w. they are then
        shifted by the requested Q-pt (Q_center). each bin center is looped over and  a reduced 
        volume of data near the bin center is read. weights are calculated by rotating the coords
        to be aligned with uvw and then calculating a rectangular 3d butterworht function on the 
        rotated grid. the signal and error where the weights are non-neglible are avg'd using 
        the weights.
        """

        # the Q-pt in cartesian coords
        self.Q_center = np.array(Q_center)

        # process binning args
        u_length, u_width, u_bin_centers = self._get_bin_centers(u_binning)
        num_u_bins = u_bin_centers.size
        print('\n*** u binning ***')
        print('u length:',u_length)
        print('u width:',u_width)
        print('u bin centers:',u_bin_centers)
        print('num u bins:',num_u_bins)

        v_length, v_width, v_bin_centers = self._get_bin_centers(v_binning)
        num_v_bins = v_bin_centers.size
        print('\n*** v binning ***')
        print('v length:',v_length)
        print('v width:',v_width)
        print('v bin centers:',v_bin_centers)
        print('num v bins:',num_v_bins)

        w_length, w_width, w_bin_centers = self._get_bin_centers(w_binning)
        num_w_bins = w_bin_centers.size
        print('\n*** w binning ***')
        print('w length:',w_length)
        print('w width:',w_width)
        print('w bin centers:',w_bin_centers)
        print('num w bins:',num_w_bins)

        self.u_bin_centers = u_bin_centers
        self.v_bin_centers = v_bin_centers
        self.w_bin_centers = w_bin_centers

        # bin widths along each axis
        self.bin_widths = np.array([u_width,v_width,w_width])

        max_length = np.array([u_length,v_length,w_length]).max()
        max_width = self.bin_widths.max()
        print('')
        print('max length:',max_length)
        print('max width:',max_width)
    
        self.grid_shape = [num_u_bins,num_v_bins,num_w_bins]

        # get mesh of steps along u,v,w
        x, y, z = np.meshgrid(u_bin_centers,v_bin_centers,w_bin_centers,indexing='ij')
        x = x.flatten(); y = y.flatten(); z = z.flatten()
        self.num_bins = x.size

        # get mesh of steps in uvw coords
        x = np.tile(x.reshape(self.num_bins,1),reps=(1,3))
        y = np.tile(y.reshape(self.num_bins,1),reps=(1,3))
        z = np.tile(z.reshape(self.num_bins,1),reps=(1,3))
        u = np.tile(self.uvw[:,0].reshape(1,3),reps=(self.num_bins,1))
        v = np.tile(self.uvw[:,1].reshape(1,3),reps=(self.num_bins,1))
        w = np.tile(self.uvw[:,2].reshape(1,3),reps=(self.num_bins,1))
        self.bin_center_coords = x*u+y*v+z*w

        # shift steps to be centered on Q_center
        Q_center_tiled = np.tile(self.Q_center.reshape(1,3),reps=(self.num_bins,1))
        self.bin_center_coords += Q_center_tiled

        # integrated data
        self.integrated_signal = np.zeros(self.num_bins)
        self.integrated_error = np.zeros(self.num_bins)

        # split bins over procs
        print('\n*** multiprocessing ***')
        flag = False
        while True:
            self.bin_inds_on_procs = np.array_split(np.arange(self.num_bins),num_procs)
            for inds in self.bin_inds_on_procs:
                if inds.size == 0: # if any procs are empty, reduce num_procs and try again
                    flag = True
                    num_procs -= 1
                    continue
            break
        if flag:
            print(f'one or more procs would calc. 0 bins. using {num_procs} procs instead')
        print(f'splitting {self.num_bins} bins over {num_procs} procs\n')

        # --- integrate the data using multiprocessing parallelism --- 

        _loop_timer = c_timer('integrate in parallel')

        self.queue = mp.Queue()
        
        _procs = []
        for _proc in range(num_procs):
            _procs.append(mp.Process(target=self._proc_loop_over_bins,args=[_proc]))

        # start execution
        for _proc in _procs:
            _proc.start()

        # get the results from queue
        for _proc in range(num_procs):

            proc, proc_integrated_signal, proc_integrated_error = self.queue.get()
            bin_inds = self.bin_inds_on_procs[proc]
            self.integrated_signal[bin_inds] = proc_integrated_signal[:]
            self.integrated_error[bin_inds] = proc_integrated_error[:]

        # close queue, kill it
        self.queue.close()
        self.queue.join_thread()

        # blocking; wait for all procs to finish before moving on
        for _proc in _procs:
            _proc.join()
            _proc.close()

        _loop_timer.stop()

    # ----------------------------------------------------------------------------------------------

    def _proc_loop_over_bins(self,proc=0):

        proc_bin_inds = self.bin_inds_on_procs[proc]
        num_proc_bins = proc_bin_inds.size
        
        R = self.rotation_matrix
        w = self.bin_widths

        proc_integrated_signal = np.zeros(num_proc_bins,dtype=float)
        proc_integrated_error = np.zeros(num_proc_bins,dtype=float)

        if proc == 0:
            print('\n*** integrating ***')
            print('proc-0 will treat >= the number of bins on other procs.\n' \
                  'only printing for proc-0.')
            print(f'there are {num_proc_bins} on proc-0\n')

        for ii, ind in enumerate(proc_bin_inds):

            if proc == 0:   
                _t = c_timer(f'bin[{ii}]')
            
            # this bin center
            Q = self.bin_center_coords[ind,:]

            # read the volume around bin center from the file
            self._read_data_from_hdf5_file(Q,delta=np.sqrt(3)*w.max())
            if proc == 0:
                print('\nshape:',self.shape)

            # get cartesian coords in rotated frame
            self._get_rotated_cartesian_coords()
            xp = self.Qxp; yp = self.Qyp; zp = self.Qzp # coords in rotated frame

            # need bin center in rotated frame too: Q' = R(Q-c) = R@Q - R@c where c is center
            Q_rot = R@Q

            # approximate orthorhombic step function
            # xp, yp, zp are rotated coords, w are widths, and Q_rot is the offset in rot. frame
            weights = self.butterworth_3d(self.Qxp,self.Qyp,self.Qzp,
                                w[0],w[1],w[2],Q_rot[0],Q_rot[1],Q_rot[2])

            bin_sig, bin_err = self._integrate_data(weights,proc=proc) # integrate the data using the weights

            proc_integrated_signal[ii] = bin_sig
            proc_integrated_error[ii] = bin_err

            if proc == 0:
                _t.stop()

        self.queue.put([proc,proc_integrated_signal,proc_integrated_error])

    # ----------------------------------------------------------------------------------------------
    
    def plot_volume(self,Q_bin_center,u_binning,v_binning,w_binning,Q_plot_center=None,scale=2e4):
        """
        read in volume to be intergrated and plot the binning region
        """

        from mayavi import mlab

        # the Q-pt in cartesian coords
        self.Q_bin_center = np.array(Q_bin_center)
        if Q_plot_center is None:
            self.Q_plot_center = self.Q_bin_center
        else:
            self.Q_plot_center = np.array(Q_plot_center)

        # process binning args
        _num_integrated_dims = 0

        u_length, u_width, u_bin_centers = self._get_bin_centers(u_binning)
        num_u_bins = u_bin_centers.size
        if num_u_bins == 1:
            _num_integrated_dims += 1
        if u_length > 0:
            u_plot_width = u_length
        else:
            u_plot_width = u_width
        print('\n*** u binning ***')
        print('u width:',u_plot_width)

        v_length, v_width, v_bin_centers = self._get_bin_centers(v_binning)
        num_v_bins = v_bin_centers.size
        if num_v_bins == 1:
            _num_integrated_dims += 1
        if v_length > 0:
            v_plot_width = v_length
        else:
            v_plot_width = v_width
        print('\n*** v binning ***')
        print('v width:',v_plot_width)

        w_length, w_width, w_bin_centers = self._get_bin_centers(w_binning)
        num_w_bins = w_bin_centers.size
        if num_w_bins == 1:
            _num_integrated_dims += 1
        if w_length > 0:
            w_plot_width = w_length
        else:
            w_plot_width = w_width
        print('\n*** w binning ***')
        print('w width:',w_plot_width)

        # we don't want to slice ALL the data for each bin, so we pad by a 
        # generous amount and slice that only that reduced volume
        max_length = np.array([u_length,v_length,w_length]).max()
        max_width = np.array([u_width,v_width,w_width]).max()
        if _num_integrated_dims == 0: # cube
            self.pad_length = (np.sqrt(3)*max_length)/2+max_width
        elif _num_integrated_dims == 1: # square 
            self.pad_length = (np.sqrt(2)*max_length)/2+max_width
        elif _num_integrated_dims == 2: # line
            self.pad_length = max_length/2+max_width
        else: # point
            self.pad_length = max_width*2
        print('\npad length:',self.pad_length)

        # bin widths along each axis
        self.bin_widths = np.array([u_width,v_width,w_width])

        # read the volume around Q_center from the file
        self._read_data_from_hdf5_file(self.Q_plot_center,self.pad_length)
        print('shape:',self.shape)

        # get cartesian coords in rotated frame
        self._get_rotated_cartesian_coords()

        # loop over bin centers and integrate the data
        R = self.rotation_matrix

        # need bin center in rotated frame too
        Q_rot = R@self.Q_bin_center

        # approximate orthorhombic step function
        xp = self.Qxp; yp = self.Qyp; zp = self.Qzp # coords in rotated frame
        weights = self.butterworth_3d(xp,yp,zp,
                u_plot_width,v_plot_width,w_plot_width,Q_rot[0],Q_rot[1],Q_rot[2])

        fig = mlab.figure(1, bgcolor=(1,1,1), fgcolor=(0,0,0),size=(500, 500))
        mlab.clf()

        x = self.Qx; y = self.Qy; z = self.Qz
        extent = [x.min(),x.max(),y.min(),y.max(),z.min(),z.max()]

        # need to mask nans to plot w/ mayavi
        signal = np.nan_to_num(self.signal,nan=0.0,posinf=0.0,neginf=0.0)*scale

        contours = []
        for ii in np.linspace(0.15,0.3,150):
            contours.append(ii)
        mlab.contour3d(x,y,z,signal,contours=contours,color=(1,0.5,1),  
                transparent=True,opacity=0.005,figure=fig)
        contours = []
        for ii in np.linspace(0.25,0.5,150):
            contours.append(ii)
        mlab.contour3d(x,y,z,signal,contours=contours,color=(1,0.75,0),
                transparent=True,opacity=0.05,figure=fig)
        contours = []
        for ii in np.linspace(0.5,10,50):
            contours.append(ii)
        mlab.contour3d(x,y,z,signal,contours=contours,color=(1,0.75,0),
            transparent=True,opacity=1.0,figure=fig)

        contours = []
        for ii in np.linspace(0.05,0.95,100):
            contours.append(ii)
        mlab.contour3d(x,y,z,weights,contours=contours,color=(0,0,1),
                transparent=True,opacity=0.075,figure=fig)

        mlab.outline(color=(0,0,0),line_width=1,extent=extent)
        mlab.axes(color=(0,0,0),line_width=1,nb_labels=5,extent=extent,
                xlabel=r'Q$_x$ (1/A)',
                ylabel=r'Q$_y$ (1/A)',
                zlabel=r'Q$_z$ (1/A)')

        mlab.orientation_axes()
        fig.scene.parallel_projection = True
        mlab.show()

    # ----------------------------------------------------------------------------------------------

    def _integrate_data(self,weights,proc=0,cutoff=1e-3):
        """
        find the coords in the volumetric data where weights function is > cutoff and integrate those
        data

        w = weights normalized to sum_i w_i = 1
        sig = sum_i [ w_i * sig_i) ] 
        err = sqrt( sum_i [ w_i * err_i ]**2 ) 
        """

        weights = weights.flatten()

        all_bins = weights.size
        inds = np.flatnonzero(weights >= cutoff)
        non_empty = inds.size

        if proc == 0:
            print('all bins:',all_bins)
            print('non empty bins:',non_empty)

        if non_empty == 0:
            return np.nan, np.nan

        weights = weights[inds]
        weights /= weights.sum()

        sig = self.signal.flatten()[inds]
        err = self.error.flatten()[inds]

        sig = (sig*weights).sum()
        err = np.sqrt(((err*weights)**2).sum())

        return sig, err

    # ----------------------------------------------------------------------------------------------

    def _get_bin_centers(self,bins):
        """
        """
        if isinstance(bins,list):
            if len(bins) == 1:
                bins = bins[0]

        if isinstance(bins,list):
            bin_range = bins[2]-bins[0]
            bin_width = bins[1]
            bins = np.round(np.arange(bins[0],bins[2]+bin_width,bin_width),4)
        else:
            bin_range = 0.0
            bin_width = bins
            bins = np.array([0.0])

        return bin_range, bin_width, bins

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

def get_uvw(which='k1'):

    """
    return uvw vectors (cartesian coords) that point along the rods. see the 'rod labels' figure 
    for which rod is which.
    """

    if which == 'k1':
        u = [ 0.221965180, 0.710773850, 0.667094360]
        v = [ 0.000000000,-0.684347171, 0.729156327]
        w = [ 0.975041521,-0.161889178,-0.151940533]

    elif which == 'k2':
        u = [-0.221965180, 0.710773850, 0.667094360]
        v = [ 0.000000000, 0.684347171,-0.729156327]
        w = [-0.975041521,-0.161889178,-0.151940533]

    elif which == 'k3':
        u = [ 0.221965180,-0.710773850, 0.667094360]
        v = [ 0.000000000,-0.684347171,-0.729156327]
        w = [ 0.975041521, 0.161889178,-0.151940533]

    elif which == 'k4':
        u = [-0.221965180,-0.710773850, 0.667094360]
        v = [ 0.000000000, 0.684347171, 0.729156327]
        w = [-0.975041521, 0.161889178,-0.151940533]

    elif which == 'h1':
        u = [ 0.710773850, 0.221965180, 0.667094360]
        v = [ 0.684347171, 0.000000000,-0.729156327]
        w = [-0.161889178, 0.975041521,-0.151940533]

    elif which == 'h2':
        u = [ 0.710773850,-0.221965180, 0.667094360]
        v = [-0.684347171, 0.000000000, 0.729156327]
        w = [-0.161889178,-0.975041521,-0.151940533]

    elif which == 'h3':
        u = [-0.710773850, 0.221965180, 0.667094360]
        v = [ 0.684347171, 0.000000000, 0.729156327]
        w = [ 0.161889178, 0.975041521,-0.151940533]

    elif which == 'h4':
        u = [-0.710773850,-0.221965180, 0.667094360]
        v = [-0.684347171, 0.000000000,-0.729156327]
        w = [ 0.161889178,-0.975041521,-0.151940533]

    else:
        print('unknown binning. ask for k1, k2, k3, k4, h1, h2, h3, or h4')
        print('returning identity matrix')
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        w = [0.0, 0.0, 1.0]

    return u, v, w

    # ----------------------------------------------------------------------------------------------

