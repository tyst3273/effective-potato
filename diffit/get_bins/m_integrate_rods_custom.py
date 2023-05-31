
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

    def __init__(self,file_name,u,v,w,subtracted=False):
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

    def _read_data_from_hdf5_file(self):
        """
        get coords from file and convert to cartesian (1/A)
        """

        # down sampling the data for speed right now
        with h5py.File(self.file_name,'r') as db:

            H = np.round(db['H'][...],6)
            K = np.round(db['K'][...],6)
            L = np.round(db['L'][...],6)

        Qx = H*self.a
        Qy = K*self.a
        Qz = L*self.c

        self.shape = [Qx.size,Qy.size,Qz.size]

        self.Qx, self.Qy, self.Qz = np.meshgrid(Qx,Qy,Qz,indexing='ij')

        H, K, L = np.meshgrid(H,K,L,indexing='ij')
        self.H = H.flatten(); self.K = K.flatten(); self.L = L.flatten()

    # ----------------------------------------------------------------------------------------------

    def _get_inds(self,Q_arr,Q,delta):
        return np.intersect1d(np.flatnonzero(Q_arr >= Q-delta),
                              np.flatnonzero(Q_arr <= Q+delta))

    # ----------------------------------------------------------------------------------------------

    def butterworth_3d(self,x,y,z,wx,wy,wz,cx=0,cy=0,cz=0,n=100):
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

        # read the volume around bin center from the file
        self._read_data_from_hdf5_file()
        print('\nshape:',self.shape)

        _loop_timer = c_timer('integrate in parallel')

        self.queue = mp.Queue()

        inds_in_bins = [[] for _ in range(self.num_bins)]
        weights_in_bins = [[] for _ in range(self.num_bins)]
        
        _procs = []
        for _proc in range(num_procs):
            _procs.append(mp.Process(target=self._proc_loop_over_bins,args=[_proc]))

        # start execution
        for _proc in _procs:
            _proc.start()

        # get the results from queue
        for _proc in range(num_procs):

            proc, inds, weights = self.queue.get()
            bin_inds = self.bin_inds_on_procs[proc]

            for ii, ind in enumerate(bin_inds):
                inds_in_bins[ind] = inds[ii]
                weights_in_bins[ind] = weights[ii]

        # close queue, kill it
        self.queue.close()
        self.queue.join_thread()

        # blocking; wait for all procs to finish before moving on
        for _proc in _procs:
            _proc.join()
            _proc.close()

        _loop_timer.stop()

        # these tell us indices and weights of Qpts in each bin
        self.inds_in_bins = inds_in_bins
        self.weights_in_bins = weights_in_bins
        
    # ----------------------------------------------------------------------------------------------

    def _proc_loop_over_bins(self,proc=0):

        proc_bin_inds = self.bin_inds_on_procs[proc]
        num_proc_bins = proc_bin_inds.size
        
        R = self.rotation_matrix
        w = self.bin_widths

        if proc == 0:
            print('\n*** integrating ***')
            print('proc-0 will treat >= the number of bins on other procs.\n' \
                  'only printing for proc-0.')
            print(f'there are {num_proc_bins} on proc-0\n')

        inds_in_bins = []
        weights_in_bins = []

        for ii, ind in enumerate(proc_bin_inds):

            if proc == 0:   
                _t = c_timer(f'bin[{ii}]')
            
            # this bin center
            Q = self.bin_center_coords[ind,:]

            # get cartesian coords in rotated frame
            self._get_rotated_cartesian_coords()
            xp = self.Qxp; yp = self.Qyp; zp = self.Qzp # coords in rotated frame

            # need bin center in rotated frame too: Q' = R(Q-c) = R@Q - R@c where c is center
            Q_rot = R@Q

            # approximate orthorhombic step function
            # xp, yp, zp are rotated coords, w are widths, and Q_rot is the offset in rot. frame
            weights = self.butterworth_3d(self.Qxp,self.Qyp,self.Qzp,
                                w[0],w[1],w[2],Q_rot[0],Q_rot[1],Q_rot[2])

            inds_in_bin, weights_in_bin = self._integrate_data(weights,proc=proc) # integrate the data using the weights
            inds_in_bins.append(inds_in_bin)
            weights_in_bins.append(weights_in_bin) 

            if proc == 0:
                _t.stop()

        self.queue.put([proc,inds_in_bins,weights_in_bins])

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

        weights = weights[inds]
        weights /= weights.sum()

        return inds, weights

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


if __name__ == '__main__':

    file_name = 'full_grid_STRUFACS.hdf5'
    
    u = [ 0.221965180, 0.710773850, 0.667094360]
    v = [ 0.000000000,-0.684347171, 0.729156327]
    w = [ 0.975041521,-0.161889178,-0.151940533]

    u_binning = [-1,0.05,1] # 1/A 
    v_binning = 0.15 # 1/A
    w_binning = 0.15 # 1/A

    c = c_integrate_rods(file_name,u,v,w)
    c.integrate([0,0,0],u_binning,v_binning,w_binning,num_procs=16)

    num_bins = c.num_bins
    bin_centers = c.u_bin_centers
    inds_in_bins = c.inds_in_bins
    weights_in_bins = c.weights_in_bins
    H = c.H; K = c.K; L = c.L

    with h5py.File('k1_u_binning.hdf5','w') as db:

        db.create_dataset('bin_centers',data=bin_centers)

        for ii in range(num_bins):

            inds = inds_in_bins[ii]
            weights = weights_in_bins[ii]

            Q_in_bin = np.zeros((inds.size,3))
            Q_in_bin[:,0] = H[inds]; Q_in_bin[:,1] = K[inds]; Q_in_bin[:,2] = L[inds]

            db.create_dataset(f'Q_in_bin_{ii}',data=Q_in_bin)
            db.create_dataset(f'weights_in_bin_{ii}',data=weights)



