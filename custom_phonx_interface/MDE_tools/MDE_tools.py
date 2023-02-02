
"""
Author: Tyler C. Sterling
Email: ty.sterling@colorado.edu
Affil: University of Colorado Boulder, Raman Spectroscopy and Neutron Scattering Lab
Date: 01/24/2022
Description:
    tools to programatically get data from mantid MDE files (in nexus format) using mantid
"""

from timeit import default_timer
import numpy as np
import h5py 
import shutil
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

    def __init__(self,MDE_file_name=None):
        """
        class to handle everything to reduce MDE => binned histo => textfile
        the point is using mantid is slow as hell with phonon expolorer so we want to 
        convert MDE to a 'database' that can be parsed quickly with SQL...
        """

        self.import_mantid()
        
        if MDE_file_name.endswith('.nxs'):
            self.MDE_ws_name = MDE_file_name[:-len('.nxs')].strip('./')
        else:
            _type = MDE_file_name.split('.')[-1].strip()
            msg = f'unknown filetype \'{_type}\'. use a *.nxs file!\n'
            crash(msg) 

        self.V_ws_name = 'V_ws'
        self.histo_ws_name = 'histo_ws'

        if MDE_file_name is not None:
            self.load_MDE(MDE_file_name)

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
            crash(err_msg,_ex)
        self.msi = msi

    # ----------------------------------------------------------------------------------------------

    def load_MDE(self,file_name):
        """
        load the MDE workspace
        """
        _t = c_timer('load_MDE',units='m')
        check_file(file_name)
        self.load_ws(file_name=file_name,ws_name=self.MDE_ws_name)
        self.print_dimensions(self.MDE_ws_name)
        self.get_lattice(self.MDE_ws_name)
        _t.stop()

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
            msg = f'\nloading workspace \'{ws_name}\' from file \'{file_name}\'\n'
            print(msg)
            self.msi.Load(Filename=file_name,OutputWorkspace=ws_name)       
        else:
            msg = f'\nworkspace \'{ws_name}\' is already loaded! continuing...\n'
            print(msg)

    # ----------------------------------------------------------------------------------------------

    def print_dimensions(self,ws_name=None):
        """
        print the dimensions in the workspace
        """
        if ws_name is None:
            ws_name = self.MDE_ws_name

        ws = self.get_ws(ws_name)
        ndims = ws.getNumDims()

        dim_names = np.zeros(ndims,dtype=object)
        dim_mins = np.zeros(ndims)
        dim_maxs = np.zeros(ndims)
        dim_units = np.zeros(ndims,dtype=object)
        dim_num_bins = np.zeros(ndims,dtype=int)

        msg = f'dimensions in workspace \'{ws_name}\':\n'
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
        print(msg+'\n')

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
            crash(msg)

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
        msg = f'lattice parameters from workspace \'{ws_name}\'\n'
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

    def bin_MDE_chunks(self,H_bins,K_bins,L_bins,E_bins,num_Q_mesh=[1,1,1],
        merged_file_name='merged_sparse_histo.hdf5',u=[1,0,0],v=[0,1,0],w=None):
        """
        split requested binning into chuncks and bin over small chunks separately. merge the 
        results of them all into a single file
        """
        
        _t = c_timer('bin_MDE_chunks',units='m')

        self.merged_file_name = merged_file_name
        if os.path.exists(merged_file_name):
            msg = 'merged file alread exists. removing it ...\n'
            print(msg)
            os.remove(merged_file_name)
        
        self.u_chunks = u; self.v_chunks = v; self.w_chunks = w
    
        # get bin edges; needed to for my algorithm to partion array into chunks
        self.H_bins = H_bins
        self.H_edges, self.H_range, self.dH = self._get_bin_edges(H_bins)
        self.K_bins = K_bins
        self.K_edges, self.K_range, self.dK = self._get_bin_edges(K_bins)
        self.L_bins = L_bins
        self.L_edges, self.L_range, self.dL = self._get_bin_edges(L_bins)

        print(self.H_edges)
        print(self.K_edges)
        print(self.L_edges)

        self.E_bins = E_bins
        self.E_edges, self.E_range, self.dE = self._get_bin_edges(E_bins)
        self.E_bins[0] -= self.E_bins[1]/2
        self.E_bins[2] += self.E_bins[1]/2

        # split bin edges (well actually bin list args) into chunks
        self.num_Q_mesh = num_Q_mesh        
        self._split_bins_on_chunks()

        # --- DEV ---
        crash()
        # -----------

        # get the data for each voxel
        self._loop_over_chunks()

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _loop_over_chunks(self):
        """
        loop over the bin edges and cut the data
        """

        msg = f'\nsplitting binning into {self.num_grid} voxels\n'
        print(msg)

        count = 0
        # loop over grid voxels
        for ii, H_bins in enumerate(self.H_chunk_bins):
            for ii, K_bins in enumerate(self.K_chunk_bins):
                for ii, L_bins in enumerate(self.L_chunk_bins):

                    _vt = c_timer(f'chunk[{count}]')

                    # bining per chunk 
                    msg = '-----------------------------------------------------------------'
                    msg += f'\nnow on grid voxel {count} spanning '
                    msg += f'\n H: {H_bins[0]:.3f} => {H_bins[2]:.3f},'
                    msg += f'  K: {K_bins[0]:.3f} => {K_bins[2]:.3f},'
                    msg += f'  L: {L_bins[0]:.3f} => {L_bins[2]:.3f}\n'
                    print(msg)

                    # get the data for this voxel
                    self.bin_MDE(H_bins,K_bins,L_bins,self.E_bins,
                            self.u_chunks,self.v_chunks,self.w_chunks)

                    # append all the non-empty bins to the file
                    self.append_sparse_to_hdf5(self.merged_file_name)

                    count += 1

                    _vt.stop()
                    print('')

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
    
    def append_sparse_to_hdf5(self,output_file_name):
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

    def _create_datasets(self,db):
        """
        create datasets in file if it doesnt already exits
        """

        _nE = self.dims[3].size

        # see below... i used 'chunks=True' to let h5py estimate these sizes
        _chunks = (125,12) # chunk size nQxnE arrays (resizable along 1st dim)
        _flat_chunks = (1000,) # chunk size for nQ arrays (resizable)

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

        db.create_dataset('Dim_0_name',data=self.dim_names[0])
        db.create_dataset('Dim_1_name',data=self.dim_names[1])
        db.create_dataset('Dim_2_name',data=self.dim_names[2])
        db.create_dataset('Dim_3_name',data=self.dim_names[3])

        db.create_dataset('H_bins',data=np.array([self.H_range[0],self.dH,self.H_range[1]]))
        db.create_dataset('K_bins',data=np.array([self.K_range[0],self.dK,self.K_range[1]]))
        db.create_dataset('L_bins',data=np.array([self.L_range[0],self.dL,self.L_range[1]]))
        db.create_dataset('E_bins',data=np.array([self.E_range[0],self.dE,self.E_range[1]]))
        db.create_dataset('DeltaE',data=self.dims[3],dtype=np.float32)

        # resizable arrays
        db.create_dataset('H',data=self.Q_mesh[0],
                                maxshape=(None,),dtype=np.float32,chunks=_flat_chunks)
        db.create_dataset('K',data=self.Q_mesh[1],
                                maxshape=(None,),dtype=np.float32,chunks=_flat_chunks)
        db.create_dataset('L',data=self.Q_mesh[2],
                                maxshape=(None,),dtype=np.float32,chunks=_flat_chunks)
        db.create_dataset('signal',data=self.signal,
                                maxshape=(None,_nE),dtype=np.float64,chunks=_chunks)
        db.create_dataset('error',data=self.err,
                                maxshape=(None,_nE),dtype=np.float64,chunks=_chunks)

    # ----------------------------------------------------------------------------------------------

    def _append_datasets(self,db):
        """
        append datasets to file if it already exists
        """

        _nE = self.dims[3].size
        _nQ = self.Q_mesh[0].size
        _nfile = db['signal'].shape[0]
        msg = f'resizing arrays from ({_nfile},{_nE}) to ({_nQ+_nfile},{_nE})\n'
        print(msg)

        db['H'].resize(_nQ+_nfile,axis=0)
        db['H'][_nfile:] = self.Q_mesh[0][...]
        db['K'].resize(_nQ+_nfile,axis=0)
        db['K'][_nfile:] = self.Q_mesh[1][...]
        db['L'].resize(_nQ+_nfile,axis=0)
        db['L'][_nfile:] = self.Q_mesh[2][...]
        db['signal'].resize(_nQ+_nfile,axis=0)
        db['signal'][_nfile:,:] = self.signal[...]
        db['error'].resize(_nQ+_nfile,axis=0)
        db['error'][_nfile:,:] = self.err[...]

    # ----------------------------------------------------------------------------------------------

    def get_sparse_arrays_from_histo_ws(self,zero_nans=False):
        """
        remove all (Q-point) bins with 0 events
        """

        _t = c_timer('sparsify')

        msg = '\nsparsifying...'
        print(msg)

        self.get_arrays_from_histo_ws()

        _shape = self.signal.shape
        _nQ = np.prod(_shape[:-1])
        _nE = _shape[-1]

        self.signal = np.reshape(self.signal,(_nQ,_nE))
        self.err = np.reshape(self.err,(_nQ,_nE))
        self.num_events = np.reshape(self.num_events,(_nQ,_nE))

        # find Q-pts where whole array (along E) is nans
        _nans = np.isnan(self.signal) # where nans are
        _infs = np.isinf(self.signal) # where +/- infs are
        _inds = ~(~_nans * ~_infs) # True for Q,E that are either nan or inf
        _inds = ~np.all(_inds,axis=1) # True for Q-pts that are all nans or infs
    
        # replace infs/nans with 0s ?
        if zero_nans:
            self.signal[(_nans)] = 0.0
            self.signal[(_infs)] = 0.0
            self.err[(_nans)] = 0.0
            self.err[(_infs)] = 0.0
            self.num_events[(_nans)] = 0.0
            self.num_events[(_infs)] = 0.0

        # strip emptys
        self.signal = self.signal[_inds,:]
        self.err = self.err[_inds,:]
        self.num_events = self.num_events[_inds,:]
        self.Q_mesh = [_Q.flatten()[_inds] for _Q in self.Q_mesh]

        # print info 
        msg = 'original shape: '+'x'.join([str(_) for _ in _shape])
        msg += f'\nnumber of Q-points: {_nQ:d}'
        msg += f'\nnumber of non-empty Q-points: {_inds.size:d}\n'
        print(msg)

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def get_arrays_from_histo_ws(self):
        """
        strip the data in the histo workspace in numpy arrays
        NOTE: see input arg NumEvNorm and effects in 'SaveMDToAscii' provided by A. Savici
        """

        self.check_ws(self.histo_ws_name)
        ws = self.get_ws(self.histo_ws_name)

        self.signal = ws.getSignalArray()
        self.err = np.sqrt(ws.getErrorSquaredArray())
        self.num_events = ws.getNumEventsArray()

        # get bin center arrays for all dimensions
        _dims = [ws.getDimension(ii) for ii in range(ws.getNumDims())]
        _dim_arrays = [self.get_dim_array(d) for d in _dims]
        _dim_mesh = self.get_dim_mesh(_dim_arrays)
        _dim_names = [d.getName() for d in _dims]
        _dim_num_bins = [d.getNBins() for d in _dims]
        _Q_mesh = self.get_dim_mesh(_dim_arrays[:-1])
        _Q_names = [d.getName() for d in _dims[:-1]]

        self.dims = _dim_arrays
        self.dim_mesh = _dim_mesh
        self.dim_names = _dim_names
        self.Q_names = _Q_names
        self.Q_mesh = _Q_mesh

    # ----------------------------------------------------------------------------------------------

    def _split_bins_on_chunks(self):
        """
        new method explicity enumerates bin edges and then grabs the 'chunks' of edges to loop
        over; old method of splitting range and then coming up with ranges didnt necessarily
        result in bins commensurate with the grid user expects. 
        """

        # number of chunks to split binning along each axis
        nH = self.num_Q_mesh[0]; nK = self.num_Q_mesh[1]; nL = self.num_Q_mesh[2]
        self.num_grid = nH*nK*nL
        self.nH = nH; self.nK = nK; self.nL = nL

        self.H_chunk_bins = self._get_chunk_bins(self.H_edges,self.dH,self.nH)
        self.K_chunk_bins = self._get_chunk_bins(self.K_edges,self.dK,self.nK)
        self.L_chunk_bins = self._get_chunk_bins(self.L_edges,self.dL,self.nL)

    # ----------------------------------------------------------------------------------------------

    def _get_chunk_bins(self,edges,d,n):
        """
        get binning range for each 'chunk' along axis
        """
        if n >= edges.size:
            msg = 'you are trying to split binning into more chunks than there are bins.\n' \
                  'pick a bigger range to bin or use fewer chunks.\n'
            crash(msg)
        _split = np.array_split(edges,n)
        chunk_bins = []
        for ii in range(n):             
            _s = _split[ii]
            # pad by d; upper bin is too big, but will never be empty (will crash if empty)
            chunk_bins.append([_s[0],d,_s[-1]+d])  
        return chunk_bins

    # ----------------------------------------------------------------------------------------------
    
    def _get_bin_edges(self,bins):
        """     
        return array of bin edges
        """
        if len(bins) != 3:
            msg = 'intergrating out dimensions is not supported.\n' \
                  'all binning args must be given as [lo, d, hi]\n'
            crash(msg)

        bin_range = [bins[0],bins[2]]
        d = bins[1]
        edges = np.arange(bin_range[0]-d/2,bin_range[1]+d,d)
        return edges, bin_range, d

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

        # copy projection to attribute the convert to str to pass to mantid
        self.u = u
        self.v = v
        self.w = w
        u = self.get_bin_str(u)
        v = self.get_bin_str(v)
        w = self.get_bin_str(w)

        # copy bin args as attributes then convert to str to pass to mantid
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

    def get_bin_str(self,bins):
        """
        get bin string arg for MDNorm
        """
        if bins is not None:
            _bins = ''
            for _ in bins:
                _bins += f'{_:.5f},'
            bins = _bins.strip(',')
        else:
            bins = ''
        return bins

    # ----------------------------------------------------------------------------------------------


    

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # temp and projection
    proj = 'parallel'

    MDE_file_name = f'../LSNO25_Ei_120meV_300K.nxs'
    out_file_name = f'LSNO25_test.hdf5'

    _t = c_timer('MDE_tools',units='m')
    
    if proj == 'parallel':
        u = [ 1, 0, 0]
        v = [ 0, 1, 0]
        w = [ 0, 0, 1]
        H_bins = [    4,   0.1,    6]
        K_bins = [    1,   0.1,    2]
        L_bins = [   -1,  0.25,    1]
        E_bins = [   10,   0.5,  100]
        num_Q_mesh = [4,4,4]
        #H_bins = [   -5,   0.1,   15]
        #K_bins = [  -12,   0.1,  7.5]
        #L_bins = [ -7.5,  0.25,  7.5]
        #E_bins = [   10,   0.5,  100]
        #num_Q_mesh = [4,4,4]

    elif proj == 'perp':
        u = [ 1, 1, 0]
        v = [ 1,-1, 0]
        w = [ 0, 0, 1]
        H_bins = [ -8.5,   0.1,  11.5]
        K_bins = [ -6.5,   0.1,  13.5]
        L_bins = [ -7.5,  0.25,   7.5]
        E_bins = [   10,   0.5,   100]
        num_Q_mesh = [4,4,4]

    # class to do the stuff
    MDE_tools = c_MDE_tools(MDE_file_name)
    MDE_tools.bin_MDE_chunks(H_bins,K_bins,L_bins,E_bins,num_Q_mesh,out_file_name,u,v,w)

    _t.stop()









