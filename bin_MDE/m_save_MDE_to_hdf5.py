
import numpy as np
import h5py
import os

# --------------------------------------------------------------------------------------------------

class _c_reduced_MDE_tools:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,MD_workspace):
        """
        reduced set of MDE tools to take WS as arg and write hdf5 file. this is a stand-alone
        class that only requires non-std libs numpy and h5py 

        IMPORTANT: this assume the workspace called 'MDNorm' as the last algorithm.
        this is because workspace history is looked up to get metadata (binning, uv, etc).
        not a huge problem if this is broken in the future, but will require fixing if it 
        is
        """
        self.MD_workspace = MD_workspace
        self.get_binning()
        self.get_lattice()
    
    # ----------------------------------------------------------------------------------------------
    
    def get_binning(self):
        """
        this might not be super robust ...
        """
        ws = self.MD_workspace

        _exp_info = ws.getExperimentInfo(0)
        _sample = _exp_info.sample()

        if not _sample.hasOrientedLattice():
            msg = '\ncouldnt find OrientedLattice object in workspace \n'
            print(msg)
            raise KeyboardInterrupt

        _uc = _sample.getOrientedLattice()
        self.u = _uc.getuVector()
        self.v = _uc.getvVector()

        _hist = ws.getHistory()
        _last_algo = _hist.lastAlgorithm()

        _algo_name = _last_algo.name()
        if not _algo_name == 'MDNorm':
            msg = f'\nshit! the last algorithm \'{_algo_name}\' is supposed to \'MDNorm\' ... \n'
            print(msg)
            raise KeyboardInterrupt

        dim_0_name = _last_algo.getPropertyValue('Dimension0Name')
        dim_0_binning = _last_algo.getPropertyValue('Dimension0Binning')
        dim_0_binning = [float(x) for x in dim_0_binning.split(',')]
        if len(dim_0_binning) == 1:
            dim_0_binning = [0,dim_0_binning[0],0]
        elif len(dim_0_binning) == 2:
            dim_0_binning = [0,dim_0_binning[1]-dim_0_binning[0],0]
        self.dim_0_binning = dim_0_binning

        dim_1_name = _last_algo.getPropertyValue('Dimension1Name')
        dim_1_binning = _last_algo.getPropertyValue('Dimension1Binning')
        dim_1_binning = [float(x) for x in dim_1_binning.split(',')]
        if len(dim_1_binning) == 1:
            dim_1_binning = [0,dim_1_binning[0],0]
        elif len(dim_1_binning) == 2:
            dim_1_binning = [0,dim_1_binning[1]-dim_1_binning[0],0]
        self.dim_1_binning = dim_1_binning

        dim_2_name = _last_algo.getPropertyValue('Dimension2Name')
        dim_2_binning = _last_algo.getPropertyValue('Dimension2Binning')
        dim_2_binning = [float(x) for x in dim_2_binning.split(',')]
        if len(dim_2_binning) == 1:
            dim_2_binning = [0,dim_2_binning[0],0]
        elif len(dim_2_binning) == 2:
            dim_2_binning = [0,dim_2_binning[1]-dim_2_binning[0],0]
        self.dim_2_binning = dim_2_binning

        dim_3_name = _last_algo.getPropertyValue('Dimension3Name')
        dim_3_binning = _last_algo.getPropertyValue('Dimension3Binning')
        dim_3_binning = [float(x) for x in dim_3_binning.split(',')]
        if len(dim_3_binning) == 1:
            dim_3_binning = [0,dim_3_binning[0],0]
        elif len(dim_3_binning) == 2:
            dim_3_binning = [0,dim_3_binning[1]-dim_3_binning[0],0]
        self.dim_3_binning = dim_3_binning

    # ----------------------------------------------------------------------------------------------

    def get_lattice(self):
        """
        get the lattice vectors for the data in the MDE file
        """
        ws = self.MD_workspace

        _exp_info = ws.getExperimentInfo(0)
        _sample = _exp_info.sample()

        if not _sample.hasOrientedLattice():
            msg = '\ncouldnt find OrientedLattice object in workspace \n'
            print(msg)
            raise KeyboardInterrupt 

        _uc = _sample.getOrientedLattice()

        # lattice vectors lenghts in Angstrom
        a = _uc.a1(); b = _uc.a2(); c = _uc.a3()
        self.a = a; self.b = b; self.c = c

        # unitcell angles in Radians
        alpha = _uc.alpha1(); beta = _uc.alpha2(); gamma = _uc.alpha3()
        self.alpha = alpha; self.beta = beta; self.gamma = gamma

        self.get_lattice_vectors_from_params()
        self.get_reciprocal_lattice_vectors()

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

    def get_cartesian_coords(self):
        """
        go from rlu to cartesian coords; vectorized method below is ~ 10x faster than looping,
        but neither are slow
        """

        _r_lat_vecs = self.recip_lattice_vectors
        _nQ = self.Q_mesh.shape[0]

        self.cartesian_Q_mesh = np.zeros((_nQ,3),dtype=float)

        for ii in range(3):
            _b = _r_lat_vecs[ii,:].reshape(1,3) # row vectors
            _b = np.tile(_b,reps=(_nQ,1))
            _Q = self.Q_mesh[:,ii].reshape(_nQ,1)
            _Q = np.tile(_Q,reps=(1,3))
            self.cartesian_Q_mesh += _Q*_b

    # ----------------------------------------------------------------------------------------------

    def get_polar_coords(self):
        """
        go from cartesian coords to polar. 0th column is magnitude of |Q|, 1st is polar angle, 2nd
        is azimuthal angle. the "z-axis" is the polar axis. 
        """

        _cart_Q = self.cartesian_Q_mesh
        _nQ = _cart_Q.shape[0]

        self.polar_Q_mesh = np.zeros((_nQ,3),dtype=float)

        # magnitude of Q
        self.polar_Q_mesh[:,0] = np.sqrt(np.sum(_cart_Q**2,axis=1))

        # mask where |Q| = 0; 
        # note, this will make the angle ~0 degrees but since |Q| ~ 0, it doesnt matter
        _mag_Q = np.copy(self.polar_Q_mesh[:,0])
        _mag_Q[np.flatnonzero(_mag_Q < 1e-6)] = 1.0

        # polar angle
        self.polar_Q_mesh[:,1] = np.arccos(_cart_Q[:,2]/_mag_Q)

        # azimuthal angle
        self.polar_Q_mesh[:,2] = np.arctan2(_cart_Q[:,1],_cart_Q[:,0])

    # ----------------------------------------------------------------------------------------------
    
    def write_sparse_hdf5(self,hdf5_file_name):
        """
        write sparse data to hdf5 file
        """

        # get the sparse data, Qpts, etc. from the workspace object
        self.get_sparse_arrays_from_histo_ws()

        # get Qpts in cartesian coords
        self.get_cartesian_coords()

        # get Qpts in polar coords of data
        self.get_polar_coords()

        msg = f'\nwriting sparse data to hdf5 file:\n  \'{hdf5_file_name}\'\n'
        print(msg)

        _exists = os.path.exists(hdf5_file_name)
        if _exists:
            msg = '\nhdf5 file already exists... removing it!\n'
            print(msg)
            os.remove(hdf5_file_name)

        with h5py.File(hdf5_file_name,mode='w',libver='latest') as db:
            self._write_datasets(db)


    # ----------------------------------------------------------------------------------------------

    def _write_datasets(self,db):
        """
        create datasets in file and write them
        """

        _nE = self.dim_arrays[3].size

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

        db.create_dataset('u',data=self.u)
        db.create_dataset('v',data=self.v)
        db.create_dataset('H_bin_args',data=np.array(self.dim_0_binning))
        db.create_dataset('K_bin_args',data=np.array(self.dim_1_binning))
        db.create_dataset('L_bin_args',data=np.array(self.dim_2_binning))
        db.create_dataset('E_bin_args',data=np.array(self.dim_3_binning))

        db.create_dataset('DeltaE',data=self.dim_arrays[3],dtype=np.float32)

        # Q rlu
        db.create_dataset('H_rlu',data=self.Q_mesh[:,0],dtype=np.float32)
        db.create_dataset('K_rlu',data=self.Q_mesh[:,1],dtype=np.float32)
        db.create_dataset('L_rlu',data=self.Q_mesh[:,2],dtype=np.float32)

        # Q cartesian
        db.create_dataset('H_cartesian',data=self.cartesian_Q_mesh[:,0],dtype=np.float32)
        db.create_dataset('K_cartesian',data=self.cartesian_Q_mesh[:,1],dtype=np.float32)
        db.create_dataset('L_cartesian',data=self.cartesian_Q_mesh[:,2],dtype=np.float32)

        # Q polar
        db.create_dataset('Q_len',data=self.polar_Q_mesh[:,0],dtype=np.float32)
        db.create_dataset('polar_angle',data=self.polar_Q_mesh[:,1],dtype=np.float32)
        db.create_dataset('azimuthal_angle',data=self.polar_Q_mesh[:,2],dtype=np.float32)

        # the data
        db.create_dataset('signal',data=self.signal,dtype=np.float64)
        db.create_dataset('error',data=self.error,dtype=np.float64)
        #db.create_dataset('num_events',data=self.num_events,dtype=int)

    # ----------------------------------------------------------------------------------------------
    
    def get_sparse_arrays_from_histo_ws(self):
        """
        remove all (Q-point) bins with only NaNs
        """

        msg = '\nsparsifying...'
        print(msg)

        self.get_arrays_from_histo_ws()

        _shape = self.signal.shape
        _nQ = np.prod(_shape[:-1])
        _nE = _shape[-1]

        self.signal = np.reshape(self.signal,(_nQ,_nE))
        self.error = np.reshape(self.error,(_nQ,_nE))
        self.num_events = np.reshape(self.num_events,(_nQ,_nE))

        # find Q-pts where whole array (along E) is nans
        _nans = np.isnan(self.signal) # where nans are
        _infs = np.isinf(self.signal) # where +/- infs are
        _inds = ~(~_nans * ~_infs) # True for Q,E that are either nan or inf
        _inds = ~np.all(_inds,axis=1) # True for Q-pts that are all nans or infs

        # strip empties
        self.signal = self.signal[_inds,:]
        self.error = self.error[_inds,:]
        self.num_events = self.num_events[_inds,:]

        # flatten Q mesh after stripping empties        
        _Q_mesh = [_Q.flatten()[_inds] for _Q in self.Q_mesh]
        self.Q_mesh = np.zeros((_Q_mesh[0].size,3),dtype=float)
        self.Q_mesh[:,0] = _Q_mesh[0][...]
        self.Q_mesh[:,1] = _Q_mesh[1][...]
        self.Q_mesh[:,2] = _Q_mesh[2][...]

        # print info 
        msg = 'original shape: '+'x'.join([str(_) for _ in _shape])
        msg += f'\nnumber of Q-points: {_nQ:d}'
        msg += f'\nnumber of non-empty Q-points: {_inds.size:d}\n'
        print(msg)

    # ----------------------------------------------------------------------------------------------
    
    def get_arrays_from_histo_ws(self):
        """
        strip the data in the histo workspace in numpy arrays
        NOTE: see input arg NumEvNorm and effects in 'SaveMDToAscii' provided by A. Savici
        """

        ws = self.MD_workspace

        self.signal = ws.getSignalArray()
        self.error = np.sqrt(ws.getErrorSquaredArray())
        self.num_events = ws.getNumEventsArray()

        # get bin center arrays for all dimensions
        _dims = [ws.getDimension(ii) for ii in range(ws.getNumDims())]
        _dim_arrays = [self.get_dim_array(d) for d in _dims]
        _dim_mesh = self.get_dim_mesh(_dim_arrays)
        _dim_names = [d.name for d in _dims]
        _dim_num_bins = [d.getNBins() for d in _dims]
        _Q_mesh = self.get_dim_mesh(_dim_arrays[:-1])
        _Q_names = [d.name for d in _dims[:-1]]

        self.dim_arrays = _dim_arrays
        self.dim_mesh = _dim_mesh
        self.dim_names = _dim_names
        self.Q_names = _Q_names
        self.Q_mesh = _Q_mesh

    # ----------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------

def save_MDE_to_hdf5(MD_workspace,hdf5_file_name):

    """
    takes MD Histogram workspace and hdf5 file name as args. gets signal, error, lattice info,
    binning, etc. from the Histogram workspace and writes it all to hdf5 file. Removes all Qpts
    that contain only NaNs along energy axis. 
    """

    MDE_tools = _c_reduced_MDE_tools(MD_workspace)
    MDE_tools.write_sparse_hdf5(hdf5_file_name)

# --------------------------------------------------------------------------------------------------


     





