import os
import numpy as np
import h5py as h5

from analysis_module.analysis_utils import _timer


class rutile_dataset:

    def __init__(self,file_name,label='bdoingbdoingbdoing'):

        """
        store data for this dataset
        """

        # scale all datasets by this; used to make vmax ~1
        self.scale = 1e5

        # lattice constants
        self.a = 4.65
        self.c = 3.0

        self.label = label
        self.file_name = os.path.abspath(file_name)
        self.file_dir = os.path.dirname(self.file_name)

        msg = '\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n' \
              f' file:\n  {self.file_name}\n label:\n  {self.label}\n'
        print(msg)


    def _check_file(self,file_name):

        """
        check if file exists
        """

        if file_name is None:
            return

        if not os.path.exists(file_name):
            msg = f'\nERROR!\n the file\n  {file_name}\n is missing!'
            exit(msg)


    def load_signal(self):

        """
        load signal, H, K, and L, and *if present* background
        """

        timer = _timer('load_file')

        with h5.File(self.file_name,'r') as in_db:
            self.H = in_db['H'][...] 
            self.K = in_db['K'][...] 
            self.L = in_db['L'][...] 
            self.signal = in_db['signal'][...]
            self.signal = np.nan_to_num(self.signal,posinf=0,neginf=0)

            if 'bg' in list(in_db.keys()):
                self.bg = in_db['bg'][...]
                self.bg = np.nan_to_num(self.bg,posinf=0,neginf=0)
        
        timer.stop()


    def get_from_nxs(self,raw_file):

        """
        strip H, K, L, and signal from the *.nxs file (raw_file) and write it to hdf5  
        """

        timer = _timer('get_from_nxs')

        raw_file = os.path.abspath(raw_file)
        self._check_file(raw_file)

        try:
            import mantid.simpleapi as msi
        except:
            msg = '\nERROR!\n' \
                  ' couldnt import mantid! make sure it is installed\n' \
                  ' and that you are in the right conda environment.'
            exit(msg)

        if not os.path.exists(self.file_dir):
            os.mkdir(self.file_dir)

        with h5.File(self.file_name,'w') as o_db:
            
            ws = msi.Load(raw_file)
            x = ws.getXDimension()
            y = ws.getYDimension()
            z = ws.getZDimension()

            # K and L dims are 3rd and 2nd inds. respectively ...
            h = np.linspace(x.getMinimum(),x.getMaximum(),x.getNBins())
            k = np.linspace(z.getMinimum(),z.getMaximum(),z.getNBins())
            l = np.linspace(y.getMinimum(),y.getMaximum(),y.getNBins())

            # i want HKL so swap K and L
            sig = np.array(ws.getSignalArray())
            sig = np.swapaxes(sig,1,2)

            o_db.create_dataset('H',h.shape)
            o_db.create_dataset('K',k.shape)
            o_db.create_dataset('L',l.shape)
            o_db['H'][...] = h[...]
            o_db['K'][...] = k[...]
            o_db['L'][...] = l[...]
            o_db.create_dataset('signal',sig.shape)
            o_db['signal'][...] = sig[...]*self.scale

        timer.stop()


    def get_bg(self,num_bins=100,Q_cut=0.07):

        """
        remove data near bragg peaks then take spherical average of intensity. this is the 
        "background" that is subtracted from the signal. it does a decent job
        """

        timer = _timer('get_bg')

        self._check_file(self.file_name)

        # --- flatten huge dataset to work with 1d array ---
        _t = _timer('flatten')
        Qx, Qy, Qz = np.meshgrid(self.H,self.K,self.L,indexing='ij')
        shape = Qx.shape
        Qdist = np.sqrt((Qx/self.a)**2+(Qy/self.a)**2+(Qz/self.c)**2)
        Qdist = Qdist.flatten()
        signal = self.signal.flatten()
        _t.stop()

        # --- get mask to "punch" out Bragg peaks ---
        _t = _timer('braggs')

        h = Qx.flatten()
        h = np.abs(np.round(np.fmod(h,1),2))
        inds = np.argwhere(h > 0.5).flatten()
        h[inds] = 1-h[inds]
        h = h.reshape(shape)

        k = Qz.flatten()
        k = np.abs(np.round(np.fmod(k,1),2))
        inds = np.argwhere(k > 0.5).flatten()
        k[inds] = 1-k[inds]
        k = k.reshape(shape)

        l = Qy.flatten()
        l = np.abs(np.round(np.fmod(l,1),2))
        inds = np.argwhere(l > 0.5).flatten()
        l[inds] = 1-l[inds]
        l = l.reshape(shape)

        mask = np.sqrt((h/self.a)**2+(k/self.a)**2+(l/self.c)**2)
        mask = (mask > Q_cut).astype(int)
        _t.stop()

        # --- histogram the data ---
        _t = _timer('histogram')
        mask = mask.flatten()

        # punch out nans and infs
        sig_min = np.nanmin(signal)
        sig_mask = -10*np.abs(sig_min)
        signal = np.nan_to_num(signal,nan=sig_mask,posinf=sig_mask,neginf=sig_mask)
        _ = np.argwhere(signal > sig_min).flatten()
        _signal = signal[_]
        _Qdist = Qdist[_]
        _mask = mask[_]
    
        # histogram the data
        inds = np.argwhere(_mask > 0).flatten() # get data thats not near bragg peaks
        counts, bins = np.histogram(_Qdist[inds],bins=num_bins) # count number of Q-points in bins
        hist, bins = np.histogram(_Qdist[inds],bins=num_bins,weights=_signal[inds]) # histogram sig 
        masked_sig = hist/counts # avg in each bin

        _t.stop()

        # --- put BG on mesh ---
        _t = _timer('mesh')
        bins = (bins[1:]+bins[:-1])/2
        inds = np.digitize(Qdist,bins[:-1],right=True)
        bg = masked_sig[inds]
        bg = bg.reshape(shape)
        Qdist = Qdist.reshape(shape)
        with h5.File(self.file_name,'r+') as o_db:
            if not 'bg' in o_db.keys():
                o_db.create_dataset('bg',shape)
                o_db.create_dataset('Qdist',shape)
            o_db['bg'][...] = bg[...]
            o_db['Qdist'][...] = Qdist[...]
        _t.stop()

        self.bg = bg

        timer.stop()


    def get_bragg_peaks(self,bragg_file=None):

        """
        create array containing all bragg peaks
        """

        if bragg_file is None:
            H_min = np.ceil(self.H.min()); H_max = np.floor(self.H.max())
            K_min = np.ceil(self.K.min()); K_max = np.floor(self.K.max())
            L_min = np.ceil(self.L.min()); L_max = np.floor(self.L.max())
            self.H_bragg = np.arange(H_min,H_max+1)
            self.K_bragg = np.arange(K_min,K_max+1)
            self.L_bragg = np.arange(L_min,L_max+1)
            _H, _K, _L = np.meshgrid(self.H_bragg,self.K_bragg,self.L_bragg,indexing='ij')
            _H = _H.flatten(); _K = _K.flatten(); _L = _L.flatten()
            self.bragg_peaks = np.array((_H,_K,_L),dtype=int).T
            """
            H_min = np.ceil(self.H.min()); H_max = np.floor(self.H.max())
            K_min = np.ceil(self.K.min()); K_max = np.floor(self.K.max())
            L_min = np.ceil(self.L.min()); L_max = np.floor(self.L.max())
            self.H_bragg = np.arange(H_min,H_max+1)
            self.K_bragg = np.arange(K_min,K_max+1)
            self.L_bragg = np.arange(L_min,L_max+1)
            self.H_bragg = np.unique(np.abs(self.H_bragg))
            self.K_bragg = np.unique(np.abs(self.K_bragg))
            self.L_bragg = np.unique(np.abs(self.L_bragg))
            _H, _K, _L = np.meshgrid(self.H_bragg,self.K_bragg,self.L_bragg,indexing='ij')
            _H = _H.flatten(); _K = _K.flatten(); _L = _L.flatten()
            self.bragg_peaks = np.array((_H,_K,_L),dtype=int).T
            """

        else: 
            _data = np.loadtxt(bragg_file,dtype=int)
            self.bragg_peaks = _data[:,:3]
            H_min = self.bragg_peaks[:,0].min(); H_max = self.bragg_peaks[:,0].max()
            K_min = self.bragg_peaks[:,1].min(); K_max = self.bragg_peaks[:,1].max()
            L_min = self.bragg_peaks[:,2].min(); L_max = self.bragg_peaks[:,2].max()
            self.H_bragg = np.arange(H_min,H_max+1)
            self.K_bragg = np.arange(K_min,K_max+1)
            self.L_bragg = np.arange(L_min,L_max+1)

            # load the 'flags' labeling 'good' brillouin zones
            if _data.shape[1] == 6:
                self.bragg_flags = _data[:,3:]


    def print_bragg_list(self,out_file='bragg_peaks'):

        """
        write list of bragg peaks in file to a text file 
        """

        self.get_bragg_peaks()
        np.savetxt(out_file,self.bragg_peaks,fmt='% 2g')


    def sum_good_braggs(self,bragg_list=None,out_file=None,flag_cut=2,bz_grid=[100,25,100],subtract_bg=False):
        
        """
        sum the data in each brillouin zone in the 'bragg_list' file. 
        """
        
        from scipy.interpolate import interpn

        eps = 0.1

        timer = _timer('sum_good_braggs')

        self._check_file(bragg_list)
        self.get_bragg_peaks(bragg_list)

        if out_file is None:
            out_file = f'{self.label}_{bragg_list}.hdf5'
        out_file = os.path.join(self.file_dir,out_file)
        print(f' writing summed bragg data to file:\n  {out_file}\n')

        #if bragg_list is not None:
        #    flags = self.bragg_flags.sum(axis=1)
        #    inds = np.flatnonzero(flags >= flag_cut-0.01)
        #else:
        #    inds = np.arange(self.bragg_peaks.shape[0])

        inds = np.arange(self.bragg_peaks.shape[0])
        peaks = self.bragg_peaks[inds,:]
        n_peaks = inds.size

        _h = np.linspace(-0.5,0.5,bz_grid[0])
        _k = np.linspace(-0.5,0.5,bz_grid[1])
        _l = np.linspace(-0.5,0.5,bz_grid[2])
        h, k, l = np.meshgrid(_h,_k,_l,indexing='ij')
        shape = h.shape
        h = h.flatten(); k = k.flatten(); l = l.flatten()

        sig = np.zeros(shape)
        count = 0

        if subtract_bg:
            signal = self.signal-self.bg
        else:
            signal = self.signal

        signal = np.nan_to_num(signal,posinf=0,neginf=0)
        
        for ii in range(n_peaks):

            h_ii = peaks[ii,0]; k_ii = peaks[ii,1]; l_ii = peaks[ii,2]
            print(f'H={h_ii:2g}',f'K={k_ii:2g}',f'L={l_ii:2g}')

            h_inds = np.arange(np.argwhere(self.H >= h_ii-(0.5+eps)).flatten()[0],
                       np.argwhere(self.H <= h_ii+(0.5+eps)).flatten()[-1])
            k_inds = np.arange(np.argwhere(self.K >= k_ii-(0.5+eps)).flatten()[0],
                       np.argwhere(self.K <= k_ii+(0.5+eps)).flatten()[-1])
            l_inds = np.arange(np.argwhere(self.L >= l_ii-(0.5+eps)).flatten()[0],
                       np.argwhere(self.L <= l_ii+(0.5+eps)).flatten()[-1])

            raw = signal[h_inds,:,:]
            raw = raw[:,k_inds,:]
            raw = raw[:,:,l_inds]

            H_ii = self.H[h_inds]; K_ii = self.K[k_inds]; L_ii = self.L[l_inds]
            grid = (H_ii,K_ii,L_ii)
            points = np.array([h+h_ii,k+k_ii,l+l_ii]).T

            try:
                _ = interpn(grid,raw,points,bounds_error=True)
            except:
                print(' - out of bounds! - ')
                continue

            sig = sig + _.reshape(shape)
            count += 1

        print('\nnumber of zones =',count)
        sig = sig/count

        with h5.File(out_file,'w') as o_db:
            o_db.create_dataset('signal',shape)
            o_db.create_dataset('h',_h.size)
            o_db.create_dataset('k',_k.size)
            o_db.create_dataset('l',_l.size)
            o_db['signal'][...] = sig[...]
            o_db['h'][:] = _h[:]
            o_db['k'][:] = _k[:]
            o_db['l'][:] = _l[:]

        timer.stop()


    def load_summed_signal(self,file_name):

        """
        load signal, h, k, and l, and summed signal on grid in 1st BZ
        """

        timer = _timer('load_summed')
        
        self.summed_file = os.path.join(self.file_dir,file_name)
        self._check_file(self.summed_file)
        
        with h5.File(self.summed_file,'r') as in_db:
            self.h = in_db['h'][...]
            self.k = in_db['k'][...]
            self.l = in_db['l'][...]
            self.summed_signal = in_db['signal'][...]

        timer.stop()




