
"""
Author: Tyler C. Sterling
Email: ty.sterling@colorado.edu
Affil: University of Colorado Boulder, Raman Spectroscopy and Neutron Scattering Lab
Date: 01/04/2022
Description:
    tools to programatically get data from mantid MDE files (in nexus format) using mantid
"""

from timeit import default_timer
import numpy as np
import h5py 
import shutil
import os


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


class c_MDE_tools:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):
        """
        class to handle everything to reduce MDE => binned histo => textfile
        the point is using mantid is slow as hell with phonon expolorer so we want to 
        convert MDE to a 'database' that can be parsed quickly with SQL...
        """

        self.import_mantid()

        self.V_ws_name = 'V_ws'
        self.MDE_ws_name = 'MDE_ws' 
        self.histo_ws_name = 'histo_ws'

        self.H_handle = '[H,0,0]'
        self.K_handle = '[0,K,0]'
        self.L_handle = '[0,0,L]'
        self.E_handle = 'DeltaE'

    # ----------------------------------------------------------------------------------------------

    def import_mantid(self):
        """
        try to import mantid. put in this method since mantid is a PITA and want control 
        over debugging errors
        """
        try:
            import mantid.simpleapi as msi
        except Exception as _ex:
            err_msg = 'couldnt import mantid! see exception below...'
            self.crash(err_msg,_ex)
        self.msi = msi

    # ----------------------------------------------------------------------------------------------

    def load_MDE(self,file_name):
        """
        load the MDE workspace
        """
        self.check_file(file_name)
        self.load_ws(file_name=file_name,ws_name=self.MDE_ws_name)
        self.get_dimensions(self.MDE_ws_name)
        self.get_lattice(self.MDE_ws_name)

    # ----------------------------------------------------------------------------------------------

    def check_ws(self,ws_name,crash=True,bonus_msg=None):
        """
        check if a workspace exists in memory on mantid backend
        if crash=True, it crashes if ws not loaded. 
        if crash=False, it just returns (bool) wheter or not its loaded 
        """
        loaded = self.msi.mtd.doesExist(ws_name)
        if not crash:
            return loaded
        if not loaded:
            msg = f'the workspace \'{ws_name}\' isnt loaded!\n'
            if bonus_msg is not None:
                msg += bonus_msg+'\n'
            self.crash(msg)

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
            msg = f'\nloading workspace \'{ws_name}\'\n'
            print(msg)
            self.msi.Load(Filename=file_name,OutputWorkspace=ws_name)       
        else:
            msg = f'\nworkspace \'{ws_name}\' is already loaded! continuing...\n'
            print(msg)

    # ----------------------------------------------------------------------------------------------

    def crash(self,err_msg=None,exception=None):
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

    # ----------------------------------------------------------------------------------------------

    def check_file(self,file_name):
        """
        check if the specified file exists
        """     
        if not os.path.exists(file_name):
            msg = f'the file:\n  \'{file_name}\' \nwas not found!'
            self.crash(msg)

    # ----------------------------------------------------------------------------------------------

    def get_dimensions(self,ws_name):
        """
        print the dimensions in the workspace
        """
        ws = self.get_ws(ws_name)
        ndims = ws.getNumDims()

        dim_names = np.zeros(ndims,dtype=object)
        dim_mins = np.zeros(ndims)
        dim_maxs = np.zeros(ndims)
        dim_units = np.zeros(ndims,dtype=object)
        dim_num_bins = np.zeros(ndims,dtype=int)
        
        msg = f'printing dimensions in workspace \'{ws_name}\'\n'
        msg += f'num_dimensions: {ndims}\n'
        msg += '--------------- (name) -------- (units) ------- (min) ---- (max) -- (bins)'
        for ii in range(ndims):
            _dim = ws.getDimension(ii)
            _min = _dim.getMinimum()
            _max = _dim.getMaximum()
            _num_bins = _dim.getNBins()
            _name = _dim.getName()
            _units = _dim.getUnits()

            dim_names[ii] = _name
            dim_mins[ii] = _min
            dim_maxs[ii] = _max
            dim_units[ii] = _units
            dim_num_bins[ii] = _num_bins

            msg += f'\ndim[{ii}]: {_name:>16s} {_units:>16s}   ' \
                   f'{_min: >10.4f} {_max: >10.4f} {_num_bins:5g}'
        print(msg)

        self.dim_names = dim_names
        self.dim_units = dim_units
        self.dim_mins = dim_mins
        self.dim_maxs = dim_maxs
        self.dim_num_bins = dim_num_bins

    # ----------------------------------------------------------------------------------------------

    def get_bin_str(self,bins):
        """
        get bin string arg for MDNorm
        """
        if bins is not None:
            _bins = ''
            for _ in bins:
                _bins += f'{_},'
            bins = _bins.strip(',')
        else:
            bins = ''
        return bins

    # ----------------------------------------------------------------------------------------------

    def get_lattice(self,ws_name):
        """
        get the lattice vectors for the data in the MDE file
        """
        ws = self.get_ws(ws_name)

        _ = ws.getExperimentInfo(0)
        _ = _.sample()

        if not _.hasOrientedLattice():
            msg = f'couldnt find OrientedLattice object in workspace \'{ws_name}\'\n'
            self.crash(msg)

        _uc = _.getOrientedLattice()
        
        # lattice vectors lenghts in Angstrom
        a = _uc.a1(); b = _uc.a2(); c = _uc.a3()
        self.a = a; self.b = b; self.c = c

        # unitcell angles in Radians
        alpha = _uc.alpha1(); beta = _uc.alpha2(); gamma = _uc.alpha3()
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

        self.get_lattice_vectors_from_params()
        self.get_reciprocal_lattice_vectors()

        # print lattice info to screen
        alpha *= 180/np.pi; beta *= 180/np.pi; gamma *= 180/np.pi
        msg = f'\nlattice parameters from workspace \'{ws_name}\'\n'
        msg += '---- a ------- b ------- c --- (Angstrom)\n'
        msg += f'{a:9.5f} {b:9.5f} {c:9.5f}\n'
        msg += '-- alpha --- beta ---- gamma - (degrees)\n'
        msg += f'{alpha:9.5f} {beta:9.5f} {gamma:9.5f}\n'
        _x = self.lattice_vectors[0,:]
        _y = self.lattice_vectors[1,:]
        _z = self.lattice_vectors[2,:]
        msg += '\n lattice vectors (Angstrom)\n'
        msg += f'{_x[0]: 9.5f} {_x[1]: 9.5f} {_x[2]: 9.5f}\n'
        msg += f'{_y[0]: 9.5f} {_y[1]: 9.5f} {_y[2]: 9.5f}\n'
        msg += f'{_z[0]: 9.5f} {_z[1]: 9.5f} {_z[2]: 9.5f}\n'
        _x = self.recip_lattice_vectors[0,:]
        _y = self.recip_lattice_vectors[1,:]
        _z = self.recip_lattice_vectors[2,:]
        msg += '\n reciprocal lattice vectors (1/Angstrom)\n'
        msg += f'{_x[0]: 9.5f} {_x[1]: 9.5f} {_x[2]: 9.5f}\n'
        msg += f'{_y[0]: 9.5f} {_y[1]: 9.5f} {_y[2]: 9.5f}\n'
        msg += f'{_z[0]: 9.5f} {_z[1]: 9.5f} {_z[2]: 9.5f}\n'
        print(msg)

    # ----------------------------------------------------------------------------------------------

    def get_lattice_vectors_from_params(self):
        """
        take lattice params (cell lenghts/angles) and return lattice vector array
        note, lengths should be in Angstrom, angles in radians
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
        recip_lattice_vectors = np.zeros((3,3),dtype=float)
        cell_vol = lattice_vectors[0,:].dot(np.cross(lattice_vectors[1,:],lattice_vectors[2,:]))
        recip_lattice_vectors[0,:] = 2*np.pi*np.cross(lattice_vectors[1,:], \
                lattice_vectors[2,:])/cell_vol
        recip_lattice_vectors[1,:] = 2*np.pi*np.cross(lattice_vectors[2,:], \
                lattice_vectors[0,:])/cell_vol
        recip_lattice_vectors[2,:] = 2*np.pi*np.cross(lattice_vectors[0,:], \
                lattice_vectors[1,:])/cell_vol
        self.recip_lattice_vectors = recip_lattice_vectors

    # ----------------------------------------------------------------------------------------------

    def bin_MDE(self,H_bins=None,K_bins=None,L_bins=None,E_bins=None,u=[1,0,0],v=[0,1,0],w=None):
        """
        bin the events in the MDE workspace into a histogram workspace; note that this doesnt 
        really return anything or create new attributes. the produced data are stored in the 
        histogram workspace
        """

        MDE_ws = self.get_ws(self.MDE_ws_name)        

        # get u, v vectors to determin axes along which to bin
        u = np.array(u); v = np.array(v)
        if w is None:
            w = np.cross(u,v)   
        else:   
            w = np.array(w)
            
        # copy projection to attribute the convert to str to pass to mantif
        self.u = u
        self.v = v
        self.w = w
        u = self.get_bin_str(u)
        v = self.get_bin_str(v)
        w = self.get_bin_str(w)
        
        # copy bin args as attributes then convert to str to pass to mantid
        self.H_bins = H_bins
        self.K_bins = K_bins
        self.L_bins = L_bins
        self.E_bins = E_bins
        H_bins = self.get_bin_str(H_bins)
        K_bins = self.get_bin_str(K_bins)
        L_bins = self.get_bin_str(L_bins)
        E_bins = self.get_bin_str(E_bins)

        msg = 'binning MDE workspace\n\n'
        msg += f'u: {u:>9}\n'
        msg += f'v: {v:>9}\n'
        msg += f'w: {w:>9}\n\n'
        msg += f'H_bins: {H_bins:>9}\n'
        msg += f'K_bins: {K_bins:>9}\n'
        msg += f'L_bins: {L_bins:>9}\n'
        msg += f'E_bins: {E_bins:>9}\n\n'
        print(msg)

        # call MDNorm to bin MDE workspace
        self.msi.MDNorm(InputWorkspace=MDE_ws,
                        QDimension0=u,
                        QDimension1=v,
                        QDimension2=w,
                        Dimension0Name='QDimension0',
                        Dimension1Name='QDimension1',
                        Dimension2Name='QDimension2',
                        Dimension3Name='DeltaE',
                        Dimension0Binning=H_bins,
                        Dimension1Binning=K_bins,
                        Dimension2Binning=L_bins,
                        Dimension3Binning=E_bins,
                        OutputWorkspace=self.histo_ws_name,
                        OutputDataWorkspace='_d',
                        OutputNormalizationWorkspace='_n')                        

    # ----------------------------------------------------------------------------------------------

    def get_dim_array(self,dim):
        """
        convert mantid dimension object to numpy array containing bin centers
        """   
        _min = dim.getMinimum()
        _max = dim.getMaximum()
        _nbins = dim.getNBins()
        _d = (_max-_min)/_nbins
        _bins = np.arange(_min+_d/2,_max,_d)
        return _bins

    # ----------------------------------------------------------------------------------------------

    def get_dim_mesh(self,dim_array_list):
        """
        take list of arrays containing bin centers and convert to an Ndim coordinate grid
        """
        return np.meshgrid(*dim_array_list,indexing='ij')

    # ----------------------------------------------------------------------------------------------

    def get_arrays_from_histo_ws(self,squeeze=False):
        """
        strip the data in the histo workspace in numpy arrays
        NOTE: see input arg NumEvNorm and effects in 'SaveMDToAscii' provided by A. Savici
        """

        self.check_ws(self.histo_ws_name)
        ws = self.get_ws(self.histo_ws_name)

        signal = ws.getSignalArray()
        err = np.sqrt(ws.getErrorSquaredArray())
        num_events = ws.getNumEventsArray()

        if squeeze:

            # note: arr.squeeze() is a numpy method that removes dimensions of size 1
            signal = signal.squeeze()
            err = err.squeeze()
            num_events = num_events.squeeze()

            # get bin center arrays for non integrated dimensions
            _dims = ws.getNonIntegratedDimensions()
            _dim_arrays = [self.get_dim_array(d) for d in _dims]
            _dim_mesh = self.get_dim_mesh(_dim_arrays)
            _dim_names = [d.getName() for d in _dims]
            _dim_num_bins = [d.getNBins() for d in _dims]

        else:
            
            # get bin center arrays for all dimensions
            _dims = [ws.getDimension(ii) for ii in range(ws.getNumDims())]
            _dim_arrays = [self.get_dim_array(d) for d in _dims]
            _dim_mesh = self.get_dim_mesh(_dim_arrays)
            _dim_names = [d.getName() for d in _dims]
            _dim_num_bins = [d.getNBins() for d in _dims]

        dims = {}
        dim_mesh = {}
        for ii, name in enumerate(_dim_names):
            dims[name] = _dim_arrays[ii]
            dim_mesh[name] = _dim_mesh[ii]

        self.signal = signal
        self.err = err
        self.num_events = num_events
        self.dims = dims
        self.dim_mesh = dim_mesh
        
    # ----------------------------------------------------------------------------------------------

    def save_histo_to_hdf5(self,output_file_name='histo.h5'):
        """
        write signal in histo workspace to hdf5 file. useful for writing colormaps, cuts, etc. 
        to file for phonon explorer, plotting, etc
        """

        _t = c_timer('save_histo_to_hdf5')

        self.get_arrays_from_histo_ws(squeeze=True)

        msg = f'\nsaving data to hdf5 file:\n  \'{output_file_name}\'\n'
        print(msg)

        with h5py.File(output_file_name,'w') as _db:
            _db.create_dataset('intensity',shape=self.signal.shape)
            _db['intensity'][...] = self.signal[...]
            _db.create_dataset('error',shape=self.err.shape)
            _db['error'][...] = self.err[...]
            _db.create_dataset('num_events',shape=self.num_events.shape)
            _db['num_events'][...] = self.num_events[...]    
            for name in self.dims.keys():
                dim = self.dims[name]
                _db.create_dataset(name,shape=dim.shape)
                _db[name][...] = dim[...]

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def get_sparse_arrays_from_histo_ws(self):
        """
        remove all (Q-point) bins with 0 events
        """

        _t = c_timer('sparsify')

        self.get_arrays_from_histo_ws(squeeze=False)

        _shape = self.signal.shape
        _nbins = self.signal.size
 
        # find empty bins
        self.num_events = self.num_events.flatten()
        _inds = np.flatnonzero(self.num_events != 0)
        _nne = _inds.size # number of non-empty bins 

        # print info 
        msg = '\nsparsifying...\n'
        msg += 'original shape: '+'x'.join([str(_) for _ in _shape])
        msg += f'\nnumber of bins: {_nbins:d}'
        msg += f'\nnumber of non-empty bins: {_nne:d}\n'
        print(msg)

        # strip emptys
        self.num_events = self.num_events[_inds]
        for name in self.dim_mesh.keys():
            self.dim_mesh[name] = self.dim_mesh[name].flatten()[_inds] 
        self.signal = self.signal.flatten()[_inds]
        self.err = self.err.flatten()[_inds]

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def save_sparse_to_hdf5(self,output_file_name='sparse.h5'):
        """
        write sparse data to hdf5 file
        """
        
        _t = c_timer('save_sparse_to_hdf5')

        # get the sparse data 
        self.get_sparse_arrays_from_histo_ws()

        msg = f'\nsaving sparse data to hdf5 file:\n  \'{output_file_name}\'\n'
        print(msg)

        with h5py.File(output_file_name,'w') as _db:
            _db.create_dataset('intensity',shape=self.signal.shape)
            _db['intensity'][...] = self.signal[...]
            _db.create_dataset('error',shape=self.err.shape)
            _db['error'][...] = self.err[...]
            _db.create_dataset('num_events',shape=self.num_events.shape)
            _db['num_events'][...] = self.num_events[...]
            for name in self.dim_mesh.keys():
                dim = self.dim_mesh[name]
                _db.create_dataset(name,shape=dim.shape)
                _db[name][...] = dim[...]

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def convert_Q_to_cartesian(self,dim_mesh):
        """
        convert Q-points from RLU to cartesian coords; since I expect there to be a HUGE 
        number of bins, this is a vectorized version. hopefully we dont run out of memory :)
        """

        _t = c_timer('Q_to_cartesian')

        _nbins = dim_mesh[list(dim_mesh.keys())[0]].shape[0]
        
        # put components into array shapes we need
        _H = dim_mesh[self.H_handle].reshape(_nbins,1)     
        _H = np.tile(_H,reps=(1,3))     
        _K = dim_mesh[self.K_handle].reshape(_nbins,1)
        _K = np.tile(_K,reps=(1,3))
        _L = dim_mesh[self.L_handle].reshape(_nbins,1)
        _L = np.tile(_L,reps=(1,3))

        # put recip. lattice vector components into shape we need
        _b1 = self.recip_lattice_vectors[0,:].reshape(1,3)
        _b1 = np.tile(_b1,reps=(_nbins,1))
        _b2 = self.recip_lattice_vectors[1,:].reshape(1,3)
        _b2 = np.tile(_b2,reps=(_nbins,1))
        _b3 = self.recip_lattice_vectors[2,:].reshape(1,3)
        _b3 = np.tile(_b3,reps=(_nbins,1))

        # Q points in cartesian coords
        Q_cart = _b1*_H+_b2*_K+_b3*_L

        # put back into dims array
        dim_mesh[self.H_handle] = Q_cart[:,0]
        dim_mesh[self.K_handle] = Q_cart[:,1]
        dim_mesh[self.L_handle] = Q_cart[:,2]

        _t.stop()
        
        return dim_mesh

    # ----------------------------------------------------------------------------------------------

    def bin_MDE_chunks(self,H_bins,K_bins,L_bins,E_bins=None,num_Q_mesh=[1,1,1],
        merged_file_name='merged_sparse_histo.txt',merged_file_format='text',
        u=[1,0,0],v=[0,1,0],w=None):

        """
        split requested binning into chuncks and bin over small chunks separately. merge the 
        results of them all into a single file
        """
        
        _t = c_timer('bin_MDE_chunks')

        self.merged_file_name = merged_file_name
        self.merged_file_format = merged_file_format
    
        # number of chunks to split binning along each axis
        nH = num_Q_mesh[0]; nK = num_Q_mesh[1]; nL = num_Q_mesh[2]
        num_grid = nH*nK*nL

        # convert bin args to arrays and save refs for later
        H_bins = np.array(H_bins)
        K_bins = np.array(K_bins)
        L_bins = np.array(L_bins)
        if H_bins.size != 3 or K_bins.size != 3 or L_bins.size != 3:
            msg = 'bin_MDE_chunks does not support explicitly integrating out Q-dimensions\n'
            self.crash(msg)
        self.H_bins_chunks = H_bins
        self.K_bins_chunks = K_bins
        self.L_bins_chunks = L_bins
        
        # binning step sizes
        self.dH = H_bins[1]
        self.dK = K_bins[1]
        self.dL = L_bins[1]
        
        # arrays of grid edges
        self.H = np.linspace(H_bins[0],H_bins[2],nH+1)
        self.K = np.linspace(K_bins[0],K_bins[2],nK+1)
        self.L = np.linspace(L_bins[0],L_bins[2],nL+1)

        _E_str = '' 
        if E_bins is not None:
            E_bins = np.array(E_bins)
            _E_str = 'E_bins: '+self.get_bin_str(E_bins)
        self.E_bins_chunks = E_bins

        # print what is planned
        _H_str = 'H_bins: '+self.get_bin_str(H_bins)
        _K_str = 'K_bins: '+self.get_bin_str(K_bins)
        _L_str = 'L_bins: '+self.get_bin_str(L_bins)
        msg = '\nbinning MDE in chunks...'
        msg += '\n'+_H_str
        msg += '\n'+_K_str
        msg += '\n'+_L_str
        msg += '\n'+_E_str
        msg += f'\n\nsplitting into a {nH:d}x{nK:d}x{nL:d} grid'
        msg += f'\nthere are {num_grid} grid voxels to do'
        msg += '\ngrid edges:'
        msg += '\nH: '+' '.join([f'{_: 4.2f}' for _ in self.H])
        msg += '\nK: '+' '.join([f'{_: 4.2f}' for _ in self.K])
        msg += '\nL: '+' '.join([f'{_: 4.2f}' for _ in self.L])
        print(msg)

        # check if file exists
        if os.path.exists(merged_file_name):    
            msg = '\nmerged file alread exists. removing it ...'
            print(msg)
            os.remove(merged_file_name)

        # this goes and cuts each voxel and writes sparse file
        self._loop_over_voxels()
        
        _t.stop()
    
    # ----------------------------------------------------------------------------------------------

    def _loop_over_voxels(self):
        """
        loop over the bin edges and cut the data
        """

        _t = c_timer('loop_over_voxels')

        count = 0
        # loop over grid voxels
        for ii in range(self.H.size-1):
            for jj in range(self.K.size-1):
                for kk in range(self.L.size-1):       
  
                    # bining per chunk 
                    H_bins = [self.H[ii],self.dH,self.H[ii+1]]
                    K_bins = [self.K[jj],self.dK,self.K[jj+1]]
                    L_bins = [self.L[kk],self.dL,self.L[kk+1]]
                    
                    msg = '\n-----------------------------------------------------------------'
                    msg += f'\nnow on grid voxel {count} spanning '
                    msg += f'\n H: {H_bins[0]:4.2f} => {H_bins[2]:4.2f},'
                    msg += f'  K: {K_bins[0]:4.2f} => {K_bins[2]:4.2f},'
                    msg += f'  L: {L_bins[0]:4.2f} => {L_bins[2]:4.2f}'
                    print(msg)

                    # get the data for this voxel
                    self.bin_MDE(H_bins,K_bins,L_bins,self.E_bins_chunks)
                    
                    # append all the non-empty bins to the file
                    if self.merged_file_format == 'text':
                        self.append_sparse_to_txt(self.merged_file_name)
                    if self.merged_file_format == 'hdf5':
                        self.append_sparse_to_hdf5(self.merged_file_name)
                    
                    count += 1 
        
        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def append_sparse_to_txt(self,output_file_name='sparse_data.txt'):
        """
        write sparse data to hdf5 file
        """

        _t = c_timer('append_sparse_to_txt')

        # get the sparse data 
        self.get_sparse_arrays_from_histo_ws()
        _has_E = (self.E_handle in self.dim_mesh)

        msg = f'\nwriting sparse data to txt file:\n  \'{output_file_name}\'\n'
        print(msg)

        # create a file header if file doesnt exist yet
        if not os.path.exists(output_file_name):               
            header = '# H[rlu] K[rlu] L[rlu] '
            _H_str = 'H_bins: '+self.get_bin_str(self.H_bins_chunks)
            _K_str = 'K_bins: '+self.get_bin_str(self.K_bins_chunks)
            _L_str = 'L_bins: '+self.get_bin_str(self.L_bins_chunks)
            _E_str = ''
            if _has_E:
                header += 'dE[meV] '
                _E_str = 'E_bins: '+self.get_bin_str(self.E_bins_chunks)
            _bin_str = '# '+', '.join([_H_str,_K_str,_L_str,_E_str])+'\n'        
            header += 'intensity error num_events\n'
            header = _bin_str+header
        else:
            header = ''

        # append all the data into a single arr to make it faster to write in the loop
        fmt = '%.2f %.2f %.2f '
        _data = np.c_[self.dim_mesh[self.H_handle],self.dim_mesh[self.K_handle],
                                                                self.dim_mesh[self.L_handle]]
        if _has_E:
            fmt += '%.2f '
            _data = np.c_[_data,self.dim_mesh[self.E_handle]]
        fmt += '%.4e %.4e %d'
        _data = np.c_[_data,self.signal,self.err,self.num_events]
       
        with open(output_file_name,'ab') as fout:
            fout.write(bytes(header,encoding='utf-8'))
            np.savetxt(fout,_data,fmt=fmt)
            
        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _create_datasets(self,db):
        """
        create datasets in file if it doesnt already exits
        """
        # see below... i used 'chunks=True' to let h5py estimate these sizes originally
        _s_chunks = (2557,) # chunk size for short floats
        _f_chunks = (2557,) # chunk size for long floats
        _i_chunks = (2557,) # chunk size for ints

        db.create_dataset('u',data=self.u)
        db.create_dataset('v',data=self.v)
        db.create_dataset('w',data=self.w)
        db.create_dataset('lattice_vectors',data=self.lattice_vectors)
        db.create_dataset('recip_lattice_vectors',data=self.recip_lattice_vectors)
        db.create_dataset('a',data=self.a)
        db.create_dataset('b',data=self.b)
        db.create_dataset('c',data=self.c)
        db.create_dataset('alpha',data=self.alpha)
        db.create_dataset('beta',data=self.beta)
        db.create_dataset('gamma',data=self.gamma)

        if self.E_handle in self.dim_mesh:
            db.create_dataset('E_bins',data=self.E_bins_chunks)
            db.create_dataset('DeltaE',data=self.dim_mesh[self.E_handle],
                                            maxshape=(None,),dtype=np.float32,chunks=_s_chunks)

        db.create_dataset('H_bins',data=self.H_bins_chunks)
        db.create_dataset('K_bins',data=self.K_bins_chunks)
        db.create_dataset('L_bins',data=self.L_bins_chunks)
        db.create_dataset('H',data=self.dim_mesh[self.H_handle],
                                    maxshape=(None,),dtype=np.float32,chunks=_s_chunks)
        db.create_dataset('K',data=self.dim_mesh[self.K_handle],
                                    maxshape=(None,),dtype=np.float32,chunks=_s_chunks)
        db.create_dataset('L',data=self.dim_mesh[self.L_handle],
                                    maxshape=(None,),dtype=np.float32,chunks=_s_chunks)
        db.create_dataset('signal',data=self.signal,
                                    maxshape=(None,),dtype=np.float64,chunks=_f_chunks)
        #db.create_dataset('error',data=self.err,
        #                            maxshape=(None,),dtype=np.float64,chunks=_f_chunks)
        db.create_dataset('num_events',data=self.num_events,
                                    maxshape=(None,),dtype=np.int32,chunks=_i_chunks)

    # ----------------------------------------------------------------------------------------------

    def _append_datasets(self,db):
        """
        append datasets to file if it already exists
        """

        _nbins = self.signal.size
        _nfile = db['signal'].size
        msg = f'resizing arrays from ({_nfile},) to ({_nbins+_nfile},)\n'
        print(msg)
        if self.E_handle in self.dim_mesh:
            db['DeltaE'].resize(_nbins+_nfile,axis=0)
            db['DeltaE'][_nfile:] = self.dim_mesh[self.E_handle][...]
        db['H'].resize(_nbins+_nfile,axis=0)
        db['H'][_nfile:] = self.dim_mesh[self.H_handle][...]
        db['K'].resize(_nbins+_nfile,axis=0)
        db['K'][_nfile:] = self.dim_mesh[self.K_handle][...]
        db['L'].resize(_nbins+_nfile,axis=0)
        db['L'][_nfile:] = self.dim_mesh[self.L_handle][...]
        db['signal'].resize(_nbins+_nfile,axis=0)
        db['signal'][_nfile:] = self.signal[...]
        #db['error'].resize(_nbins+_nfile,axis=0)
        #db['error'][_nfile:] = self.err[...]
        db['num_events'].resize(_nbins+_nfile,axis=0)
        db['num_events'][_nfile:] = self.num_events[...]

    # ----------------------------------------------------------------------------------------------
    
    def append_sparse_to_hdf5(self,output_file_name='sparse_data.h5'):
        """
        write sparse data to hdf5 file; needs to be resizable... gonna be hard
        """

        _t = c_timer('append_sparse_to_hdf5')

        # get the sparse data 
        self.get_sparse_arrays_from_histo_ws()

        msg = f'\nwriting sparse data to hdf5 file:\n  \'{output_file_name}\'\n'
        print(msg)

        _exists = os.path.exists(output_file_name)
        with h5py.File(output_file_name,'a') as db:

            # create datasets if file doesnt exist
            if not _exists:
                self._create_datasets(db)

            # if file already exists, resize and add data to datasets
            else:
                self._append_datasets(db)

        _t.stop()

    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # which file to do
    MDE_file_name = '../LSNO25_Ei_120meV_300K.nxs'

    # which macro to run
    task = 'create_sparse_hdf5'
    
    _t = c_timer(task,units='m')

    # good data range is H=-5,15; K=-12,7.5; L=-7.5,7.5
    # start, step size, end
    H_bins = [  -5, 0.1,  15]
    K_bins = [ -12, 0.1, 7.5]
    L_bins = [-7.5, 0.25, 7.5]
    E_bins = [  10, 0.5, 100]
    num_Q_mesh = [8,8,8]
    
    #H_bins = [  4,  0.1,  8]
    #K_bins = [ -2,  0.1,  2]
    #L_bins = [ -3,  0.1,  3]
    #E_bins = [ 50,  0.25, 100]
    #num_Q_mesh = [6,6,6]

    # class to do the stuff
    MDE_tools = c_MDE_tools()
    MDE_tools.load_MDE(MDE_file_name)

    # bin/normalize data and save to histogram file
    if task == 'bin_single':
        MDE_tools.bin_MDE(H_bins,K_bins,L_bins,E_bins)
        MDE_tools.save_histo_to_hdf5('single_cut.h5')

    # bin/normalize data and save to histogram file
    elif task == 'bin_single_sparse':
        MDE_tools.bin_MDE(H_bins,K_bins,L_bins,E_bins)
        MDE_tools.save_sparse_to_hdf5('single_cut_sparse.h5')

    # divide binning over large range into 'chunks'
    elif task == 'create_sparse_text':
        MDE_tools.bin_MDE_chunks(H_bins,K_bins,L_bins,E_bins,num_Q_mesh,
                    merged_file_name='LSNO25_300K.txt',merged_file_format='text')

    # diving binning over large range into 'chunks' and write to hdf5 file
    elif task == 'create_sparse_hdf5':
        MDE_tools.bin_MDE_chunks(H_bins,K_bins,L_bins,E_bins,num_Q_mesh,
                    merged_file_name='LSNO25_300K.hdf5',merged_file_format='hdf5')

    # do nothing
    else:
        msg = f'\ndunno what task \'{task}\' is!\n'
        print(msg)

    _t.stop()









