
"""
Author: Tyler C. Sterling
Email: ty.sterling@colorado.edu
Affil: University of Colorado Boulder, Raman Spectroscopy and Neutron Scattering Lab
Date: 04/26/2023
Description:
    tools to:
        - programatically get data from mantid MDE files (in nexus format) using mantid
            and write to custom hdf5 for quick access from phonon explorer
        - calculate and subtract background from raw data in custom hdf5 file
"""

from .m_file_utils import *

import numpy as np
import h5py 
import os
from scipy.signal import convolve
import multiprocessing as mp


# --------------------------------------------------------------------------------------------------
# class to calculate and subtract about background
# --------------------------------------------------------------------------------------------------
    
class c_background_tools:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,raw_file):

        """
        class that reads raw file, smooths the data in it, and then subtracts the background from
        Q-points in the raw file by doing "rocking scans" and taking the minimum of each. 
        """

        check_file(raw_file)
        self.raw_file = raw_file

        self.smoothed_file = self._append_suffix_to_file_name(raw_file,suffix='SMOOTHED')
        self.background_file = self._append_suffix_to_file_name(raw_file,suffix='BACKGROUND')
        self.background_subtracted_file = self._append_suffix_to_file_name(
                    raw_file,suffix='BACKGROUND_SUBTRACTED')

    # ----------------------------------------------------------------------------------------------

    def _append_suffix_to_file_name(self,file_name,suffix):

        """
        remove file type from file name and append a suffix to the leading text. put filetype 
        back on
        """

        file_type = file_name.split('.')[-1]
        file_type = '.'+file_type
        file_prefix = file_name[:-len(file_type)]
        new_file_name = file_prefix+'_'+suffix+file_type

        return new_file_name

    # ----------------------------------------------------------------------------------------------

    def make_smoothed_file(self,smoothing_fwhm=None,num_blocks=1):

        """
        split the Q-point set in raw file in num_blocks blocks. for each, block
        interpolate the data onto a fine energy grid and then Gaussian smooth the data. 
        """

        print('\n*** smoothing data ***\n')
        
        check_file(self.raw_file)

        with h5py.File(self.raw_file,'r') as raw_db, \
                h5py.File(self.smoothed_file,'w') as smooth_db:
           
            self.num_Q = raw_db['H_rlu'].size
            self.energy = raw_db['DeltaE'][...]

            if smoothing_fwhm is None:
                self.smoothing_fwhm = (self.energy[1]-self.energy[0])*3.0
            else:
                self.smoothing_fwhm = float(smoothing_fwhm)

            self._copy_header_datasets(raw_db,smooth_db)

            # add new header info
            smooth_db.create_dataset('smoothing_fwhm',data=self.smoothing_fwhm)

            # go and do the smoothing 
            self._smooth_by_looping_over_Q_blocks(num_blocks,raw_db,smooth_db)

    # ----------------------------------------------------------------------------------------------

    def _smooth_by_looping_over_Q_blocks(self,num_blocks,raw_db,smooth_db):

        """
        split Q into blocks and loop over the blocks, smoothing the Q-point in each block
        """

        _t = c_timer('loop_over_Q_blocks')

        Q_index_blocks = np.arange(self.num_Q)
        Q_index_blocks = np.array_split(Q_index_blocks,num_blocks)

        print(f'\nsplitting data into {num_blocks} blocks\n')

        for block in range(num_blocks):

            _bt = c_timer(f'block[{block}]')
            
            Q_indices = Q_index_blocks[block]
            signal = raw_db['signal'][Q_indices,:]
            smooth_db['signal'][Q_indices,:] = \
                    self._gaussian_smooth(self.energy,signal,self.smoothing_fwhm)

            _bt.stop()

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _copy_header_datasets(self,source_db,dest_db):

        """
        copy 'header' info from database_1 to database_2
        """

        _t = c_timer('copy_datasets')

        print('\ncopying_datasets ...\n')

        keys = list(source_db.keys())
       
        # don't want to copy raw signal. will write new data later 
        keys.pop(keys.index('signal'))
        dest_db.create_dataset('signal',shape=source_db['signal'].shape)

        for key in keys:
            dest_db.create_dataset(key,data=source_db[key])

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _gaussian_smooth(self,x,y,fwhm):

        """
        smooth data using gaussian convolution. x is a 1d array, y is a 1d array or 2d array 
        with same shape as x along 2nd axis. calls another method to actually do the smoothing
        """

        # gaussian stddev
        sigma = fwhm/np.sqrt(8*np.log(2))
        
        # step size along x axis
        dx = x[1]-x[0]

        # x axis for gaussian function
        gx = np.arange(-6*sigma,6*sigma,dx)
        gaussian = np.exp(-0.5*(gx/sigma)**2)

        if y.ndim == 1:
            smooth = self._gaussian_smooth_1d(x,y,gaussian)
        else:
            smooth = np.zeros(y.shape)
            for ii in range(smooth.shape[0]):
                smooth[ii,:] = self._gaussian_smooth_1d(x,y[ii,:],gaussian)

        return smooth

    # ----------------------------------------------------------------------------------------------

    def _gaussian_smooth_1d(self,x,y,gaussian):

        """
        smooth data using gaussian convolution. if there are nans in input array, interpolate to 
        remove the nans. 
        """

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

    def calculate_background(self,num_bg_cuts=10,delta_Q_length=0.5,delta_polar_angle=10,
                                     delta_azimuthal_angle=10,num_Q_point_procs=1):

        """
        do a 'rocking scan' around each Q-pt in the raw file and get random neighboring points 
        that lie in a piece of spherical shell the same Q-pt. 
        
        num_bg_cuts determines the number of neighboring points to use.

        the neighboring points are bounded by |Q'| in |Q| +- delta_Q_length (in 1/A) and
        angle' in angle +- delta_angle (in degrees)

        the neighboring points are read from the smoothed file and the background is taken as the
        point-by-point minimum of all of the Q-points (neighboring and original).

        the idea is that the background is spherical, while the single-crystal scattering is 
        not. so looking at data lying on the same sphere but offset by a small angle means 
        we will move away from the phonons and only look at the background. ofcourse it is 
        possible that we might find other phonons at another Q-pt... so we use multiple 
        background cuts that are randomly chosen as a fail safe.
        """
        
        print('\n*** calculating background ***\n')
        
        _t = c_timer('calculate_background')

        check_file(self.raw_file)
        check_file(self.smoothed_file)

        with h5py.File(self.smoothed_file,'r') as _db:
            self.num_Q = _db['H_rlu'].size
            self.energy = _db['DeltaE'][...]

        self.num_E = self.energy.size
        self.background_signal = np.zeros((self.num_Q,self.num_E),dtype=float)

        # split Qpt indices onto multiple process to get and calculate background 
        self.Q_indices_on_procs = np.array_split(np.arange(self.num_Q),num_Q_point_procs)
        print('\nnumber of Q-pts on each proc:')
        msg = ''
        for ii in range(num_Q_point_procs):
            _ = f'proc[{ii}]'
            msg += f'{_:10} {self.Q_indices_on_procs[ii].size}\n'
        print(msg)

        # +- bounds for the polar coordinates
        self.delta_polar_angle = delta_polar_angle*np.pi/180 # degrees => radians
        self.delta_azimuthal_angle = delta_azimuthal_angle*np.pi/180 # degrees => radians
        self.delta_Q_length = delta_Q_length

        # Queue for passing data between procs
        self.queue = mp.Queue()

        proc_list = []
        for pp in range(num_Q_point_procs):
            proc_list.append(mp.Process(target=self._get_background_by_rocking_scans,
                                args=[pp,num_bg_cuts]))

        # start execution
        for pp in proc_list:
            pp.start()

        # get the results from queue
        for pp in range(num_Q_point_procs):

            proc, background_signal_on_proc = self.queue.get()
            Q_indices = self.Q_indices_on_procs[proc]
            self.background_signal[Q_indices,:] = background_signal_on_proc[...]

        # blocking; wait for all procs to finish before moving on
        for pp in proc_list:
            pp.join()
            pp.close()

        self._write_background_file()

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _write_background_file(self):

        """
        create background file and copy datasets from raw file into it
        """

        _t = c_timer('write_background_file')

        with h5py.File(self.smoothed_file,'r') as smooth_db, \
                h5py.File(self.background_file,'w') as bg_db:

            self._copy_header_datasets(smooth_db,bg_db)
            bg_db['signal'][...] = self.background_signal[...]

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _get_background_by_rocking_scans(self,proc,num_bg_cuts):

        """
        do a 'rocking scan' around each Q-pt in the raw file and get random neighboring points
        that lie in a piece of spherical shell centered Q-pt.

        note, since Qpts are split over procs, only arg we need is which proc this is

        outline:    loop over all Qpts on this proc. 
                    for each one, randomly choose upto num_bg_cuts number of neighboring Qpts. 
                    get the smooth data for all of them 
                    calculate BG as point-by-point minimum in the smooth data
                    fill a BG array for all of the Qpts on this proc.
                    return to main proc using multiprocessing Queue 
                    main proc writes BG arrays to file
        """

        if proc == 0:
            _t = c_timer('get_background_by_rocking_scans')
            print('\nonly printing progress for proc 0\n')

        Q_indices = self.Q_indices_on_procs[proc]
        num_Q_on_proc = Q_indices.size
        background_signal = np.zeros((num_Q_on_proc,self.num_E),dtype=float)

        with h5py.File(self.raw_file,mode='r',libver='latest',swmr=True) as raw_db, \
            h5py.File(self.smoothed_file,mode='r',libver='latest',swmr=True) as smooth_db:

            # need these to find Qpt and neighbors in spherical coords
            self.Q_len = raw_db['Q_len'][...]
            self.polar_angle = raw_db['polar_angle'][...]
            self.azimuthal_angle = raw_db['azimuthal_angle'][...]

            for ii, Q_index in enumerate(Q_indices):

                num_cuts = num_bg_cuts

                if proc == 0:
                    if ii%1000 == 0:
                        print(f'  Qpt {ii} out of {num_Q_on_proc}')
                
                # get this Qpt
                Q_ii = self.Q_len[Q_index]
                polar_ii = self.polar_angle[Q_index]
                azimuthal_ii = self.azimuthal_angle[Q_index]
                
                # get neighbors
                neighbor_inds = self._get_neighboring_Q_points_in_shell(Q_ii,polar_ii,azimuthal_ii)
                num_neighbors = neighbor_inds.size

                # if no neighbors, background is 0
                if num_neighbors == 0:
                    msg = '\n*** WARNING ***\n'
                    msg += f'no bg cuts for Qpt num. {Q_index}!\n'
                    msg += 'setting bg to 0 for this Qpt\n'
                    print(msg)

                    background_signal[ii,:] = np.zeros(self.num_E,dtype=float)

                    continue

                # use all data available if num_neighbors < num_bg_cuts
                if num_neighbors < num_bg_cuts:

                    # get all neighbors if num_neighbors < num_bg_cuts
                    num_cuts = num_neighbors 


                # randomly pick neighbors
                np.random.shuffle(neighbor_inds) # shuffles in place ...
                neighbor_inds = neighbor_inds[:num_cuts]

                background_signal[ii,:] = self._calculate_background_from_pt_by_pt_min(
                        Q_index,neighbor_inds,smooth_db)

        # put in queue to return to main proc
        self.queue.put([proc,background_signal])

        if proc == 0:
            _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _calculate_background_from_pt_by_pt_min(self,Q_index,neighbor_inds,smooth_db):

        """
        cut Q_pt and neighbors from smoothed data and calculate background as the 
        point-by-point minimum of the data
        """

        _mask = 1e8
        
        neighbor_inds = np.sort(neighbor_inds)
        num_neighbors = neighbor_inds.size
        smooth_signal = np.zeros((num_neighbors+1,self.num_E),dtype=float)
        smooth_signal[0,:] = smooth_db['signal'][Q_index,:]
        smooth_signal[1:,:] = smooth_db['signal'][neighbor_inds,:]

        smooth_signal = np.nan_to_num(smooth_signal,nan=_mask,posinf=_mask,neginf=_mask)
        background_signal = np.min(smooth_signal,axis=0)

        background_signal[np.flatnonzero(~(background_signal < _mask))] = 0.0

        return background_signal

    # ----------------------------------------------------------------------------------------------

    def _get_neighboring_Q_points_in_shell(self,Q,polar,azimuthal):

        """
        find neighboring Q-pts that are bounded by |Q|+-dQ, polar+-d_polar, azi+-d_azi,
        i.e. all atoms in the shell centered on the Q-pt defined earlier
        """

        dQ = self.delta_Q_length
        dpolar = self.delta_polar_angle
        dazi = self.delta_azimuthal_angle

        # Q_len bounds
        len_inds = np.flatnonzero(self.Q_len >= Q-dQ)
        len_inds = np.intersect1d(np.flatnonzero(self.Q_len <= Q+dQ),len_inds)
            
        # polar angle bounds
        polar_inds = np.flatnonzero(self.polar_angle >= polar-dpolar)
        polar_inds = np.intersect1d(np.flatnonzero(self.polar_angle <= polar+dpolar),polar_inds)

        # polar angle bounds
        azimuthal_inds = np.flatnonzero(self.azimuthal_angle >= azimuthal-dazi)
        azimuthal_inds = np.intersect1d(
            np.flatnonzero(self.azimuthal_angle <= azimuthal+dazi),azimuthal_inds)

        # intersection of all of these is the shell
        neighbor_inds = np.intersect1d(len_inds,polar_inds)
        neighbor_inds = np.intersect1d(neighbor_inds,azimuthal_inds)

        return neighbor_inds
    
    # ----------------------------------------------------------------------------------------------

    def subtract_background(self,num_blocks=1):

        """
        open raw and bg files and subtract bg from raw data
        """
        
        print('\n*** subtracting data ***\n')
        
        _t = c_timer('subtract_background')

        with h5py.File(self.raw_file,'r') as raw_db, \
            h5py.File(self.background_file,'r') as bg_db, \
            h5py.File(self.background_subtracted_file,'w') as sub_db:

            self._copy_header_datasets(bg_db,sub_db)
            
            Q_index_blocks = np.arange(self.num_Q)
            Q_index_blocks = np.array_split(Q_index_blocks,num_blocks)

            print(f'\nsplitting data into {num_blocks} blocks\n')

            for block in range(num_blocks):
                
                Q_indices = Q_index_blocks[block]
                raw_signal = raw_db['signal'][Q_indices,:]
                bg_signal = bg_db['signal'][Q_indices,:]
                sub_db['signal'][Q_indices,:] = raw_signal-bg_signal

        _t.stop()


    # ----------------------------------------------------------------------------------------------



