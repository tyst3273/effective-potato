
from euphonic import ForceConstants, ureg
from euphonic.util import mp_grid

import matplotlib.pyplot as plt
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

class c_euphonic_sqw:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,path=None,phonopy_file='phonopy.yaml',hdf5_file=None):
        
        """
        class to interact with euphonic to get and broaden structure factors.
        if cmap_file is given, a colormap is read from the file to plot. nothing else is done.
        """

        if path is None:
            path = os.getcwd()

        if hdf5_file is not None:
            hdf5_file = os.path.join(path,hdf5_file)
            self._get_strufacs_from_file(hdf5_file)
            return

        check_file(os.path.join(path,phonopy_file))
        self.force_constants_object = ForceConstants.from_phonopy(
                path=path,summary_name=phonopy_file)

        # row vectors, convert Bohr to Angstrom
        self.lattice_vectors = self.force_constants_object.crystal._cell_vectors/1.88973 
        self.reciprocal_lattice_vectors = \
                np.linalg.solve(self.lattice_vectors,2*np.pi*np.identity(3)).T

    # ----------------------------------------------------------------------------------------------

    def _get_strufacs_from_file(self,cmap_file):

        """
        read the cmap data from the file
        """

        check_file(cmap_file)

        with h5py.File(cmap_file,'r') as db:
            self.raw_energues = db['raw_energies'][...]
            self.raw_structure_factors = db['raw_structure_factors'][...]
            if 'cmap_structure_factors' in db.keys():
                self.cmap_structure_factors = db['cmap_structure_factors'][...]
                self.cmap_energies = db['cmap_energies'][...]
                self.cmap_Qpt_distances = db['cmap_Qpt_distances'][...]

    # ----------------------------------------------------------------------------------------------

    def set_Qpts(self,Qpts):

        """
        set Qpts created externally
        """

        self.Qpts_option = 'user'

        self.Qpts = Qpts
        self.num_Qpts = Qpts.shape[0]

    # ----------------------------------------------------------------------------------------------

    def _get_bin_centers(self,bins):

        """
        get array of bin centers
        """

        if not isinstance(bins,list):
            bins = [bins]

        msg = '*_bins must be a list of floats'
        try:
            bins = np.array(bins,dtype=float)
        except:
            crash(msg)

        if bins.size == 1:
            return bins
        else:
            _n = int(((bins[2]-bins[0])/bins[1]).round())+1
            bins = np.linspace(bins[0],bins[2],_n)
            return bins

    # ----------------------------------------------------------------------------------------------

    def rotate_Qpts(self,R):

        """
        Q' = RQ 
        """

        for ii in range(self.num_Qpts):
            self.Qpts[ii,:] = R@self.Qpts[ii,:]

    # ----------------------------------------------------------------------------------------------

    def get_Qpts_on_grid(self,u_bins,v_bins,w_bins,proj=None,center=[0,0,0]):

        """
        take a set of 'vertices' and number of steps along the *shortest* segment and generate
        Qpts on path thru reciprocal space
        """

        self.Qpts_option = 'grid'

        # get bin centers 
        _u_bins = self._get_bin_centers(u_bins)
        _num_u = _u_bins.size
        print('\nnum u bins:',_num_u)
        print('u bins:',_u_bins)
        _v_bins = self._get_bin_centers(v_bins)
        _num_v = _v_bins.size
        print('num v bins:',_num_v)
        print('v bins:',_v_bins)
        _w_bins = self._get_bin_centers(w_bins)
        _num_w = _w_bins.size
        print('num w bins:',_num_w)
        print('w bins:',_w_bins,'\n')

        self.u_bins = _u_bins
        self.v_bins = _v_bins
        self.w_bins = _w_bins

        # mesh of bin centers
        _u_bins, _v_bins, _w_bins = np.meshgrid(_u_bins,_v_bins,_w_bins,indexing='ij')
        self.Qpts_grid_shape = _u_bins.shape
        _u_bins = _u_bins.flatten(); _v_bins = _v_bins.flatten(); _w_bins = _w_bins.flatten()

        self.num_Qpts = _u_bins.size
        print('num Qpts:',self.num_Qpts,'\n')

        # change basis if diff. projection is given
        if proj is None:
            self.Qpts = np.c_[_u_bins,_v_bins,_w_bins]
            self.proj = np.identity(3)
        else:
            self.Qpts = np.zeros((self.num_Qpts,3))
            proj = np.array(proj)
            self.proj = proj
            for ii in range(self.num_Qpts):
                self.Qpts[ii,:] = proj[0,:]*_u_bins[ii]+ \
                                  proj[1,:]*_v_bins[ii]+ \
                                  proj[2,:]*_w_bins[ii]

        self.center = np.array(center)
        self.Qpts[:,0] += center[0]; self.Qpts[:,1] += center[1]; self.Qpts[:,2] += center[2]

        return self.Qpts, self.num_Qpts

    # ----------------------------------------------------------------------------------------------

    def get_Qpts_on_path(self,Qpt_vertices,num_Qpt_steps):
        
        """
        take a set of 'vertices' and number of steps along the *shortest* segment and generate
        Qpts on path thru reciprocal space
        """

        self.Qpts_option = 'path'

        Qpt_vertices = np.array(Qpt_vertices)

        _num_verts = len(Qpt_vertices)-1
        _steps = np.zeros(_num_verts)
        _dist = np.zeros(_num_verts+1)

        # get lengths of segments and cumulative distance of each vertex
        for ii in range(_num_verts):
            _dQ = self.reciprocal_lattice_vectors@(Qpt_vertices[ii+1,:]-Qpt_vertices[ii,:])
            _dQ = np.sqrt(_dQ@_dQ)
            _steps[ii] = _dQ
            _dist[ii+1] = _dist[ii]+_dQ

        # convert lengths to integer number of steps
        _steps = (_steps/_steps.min()*num_Qpt_steps).round().astype(int)
        num_Qpts = _steps.sum()

        # fill Qpt and cumulative Qpt distance arrays
        Qpt_distances = np.zeros(num_Qpts)
        Qpts = np.zeros((num_Qpts,3))

        shift = 0
        for ii in range(_num_verts):
            _s = _steps[ii]
            _dQ = (_dist[ii+1]-_dist[ii+1])/_s
            Qpt_distances[shift:shift+_s] = np.linspace(_dist[ii],_dist[ii+1],_s+1)[:-1]
            for jj in range(3):
                Qpts[shift:shift+_s,jj] = np.linspace(Qpt_vertices[ii,jj],
                            Qpt_vertices[ii+1,jj],_s+1)[:-1]
            shift += _s

        self.Qpts = Qpts
        self.Qpt_distances = Qpt_distances
        self.num_Qpts = num_Qpts
       
        return Qpts, num_Qpts, Qpt_distances

    # ----------------------------------------------------------------------------------------------

    def _calculate_debye_waller(self,dw_grid,asr,dipole):

        """
        calculate debye-waller factor for later use in structure factor calculation
        """

        _t = c_timer('calculate_debye_waller')

        if not hasattr(self,'temperature'):
            crash('must define temperature to calculate debye-Waller factor!')

        dw_grid = mp_grid(dw_grid)
        dw = self.force_constants_object.calculate_qpoint_phonon_modes(
                    dw_grid,asr=asr,dipole=dipole)
        dw = dw.calculate_debye_waller(temperature=self.temperature)

        _t.stop()

        return dw

    # ----------------------------------------------------------------------------------------------

    def _plot_dispersion(self,modes):

        """
        plot the dispersions for deubugging
        """

        fig, ax = plt.subplots(figsize=(8,8))

        _d = modes.to_dict()

        if hasattr(self,'Qpts_distances'):
            _Q = self.Qpts_distances
            label = r'|Q| [$\AA^{-1}$]'
        else:
            _Q = np.arange(self.num_Qpts)
            label = 'Q-index'

        _f = _d['frequencies']
        for ii in range(_f.shape[1]):
            ax.plot(_f[:,ii],marker='o',ms=1,lw=1,ls='-',c='k')

        ax.plot([_Q.min(),_Q.max()],[0,0],lw=2,ls=':',c='k',ms=0)

        if _f.min() > 0:
            _f_min = 0
        else:
            _f_min = _f.min()*1.1

        ax.axis([_Q.min(),_Q.max(),_f_min,_f.max()*1.1])

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1,labelsize='x-large')
        ax.tick_params(which='major',length=5)
        ax.tick_params(which='minor',length=2)

        ax.set_xlabel(label,labelpad=4,fontsize='x-large')
        ax.set_ylabel('E [meV]',labelpad=4,fontsize='x-large')

        plt.show()
        plt.close()
        plt.clf()

    # ----------------------------------------------------------------------------------------------

    def calculate_structure_factors(self,asr='realspace',dipole=True,temperature=None,
                dw_grid=None,plot_dispersion=False,scattering_lengths='Sears1992'):

        """
        calculate structure factors along the Qpt path set using set_Qpts. if you want to include
        the debye waller factor, set temperature and dw_grid. 
        """

        _t = c_timer('calculate_structure_factors')

        if isinstance(scattering_lengths,dict):
            fm = ureg('fm')
            for key in scattering_lengths.keys():
                b = scattering_lengths[key]
                if ureg.Quantity(b).check(fm):
                    continue
                else:
                    scattering_lengths[key] *= fm
            print(scattering_lengths)

        _modes = self.force_constants_object.calculate_qpoint_phonon_modes(
                self.Qpts,asr=asr,dipole=dipole)

        if plot_dispersion:
            self._plot_dispersion(_modes)

        if temperature is not None:
            self.temperature = temperature*ureg('K')

        if dw_grid is None:
            self.structure_factors_object = _modes.calculate_structure_factor(scattering_lengths)
        else:
            dw = self._calculate_debye_waller(dw_grid,asr,dipole)
            self.structure_factors_object = _modes.calculate_structure_factor(dw=dw,
                    scattering_lengths=scattering_lengths)

        _sqw = self.structure_factors_object.to_dict()

        # [ num_Q , num_modes ]
        self.raw_energies = _sqw['frequencies'][...] # meV
        self.raw_structure_factors = _sqw['structure_factors'] # milliBarn 

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def get_colormap(self,E_min=-150,E_max=150,dE=0.1,E_width=2.5,Q_width=None,
                temperature=None,calc_bose=True):
        
        """
        get a colormap of the structure factors. E_width is mandatory, Q_width is optional. 
        temperature is only required if calc_bose = True. its optional if set when calculating 
        debeye waller factors. if different than the previous value, new temp is used. 
        """

        _sqw = self.structure_factors_object 
        E_bins = np.arange(E_min,E_max+dE,dE)
        #E_bins[np.flatnonzero(np.abs(E_bins) < 1e-3)] = 1e-3
        E_bins *= ureg('meV')
        E_width *= ureg('meV')

        if Q_width is not None:
            Q_width *= ureg('1/angstrom')

        if temperature is not None:
            _sqw.temperature = temperature*ureg('K')

        _cmap = _sqw.calculate_sqw_map(E_bins,calc_bose=calc_bose)
        self.colormap_object = _cmap.broaden(x_width=Q_width,y_width=E_width)
        
        _cmap = self.colormap_object.to_dict()

        self.cmap_structure_factors = _cmap['z_data'][...]
        _E = _cmap['y_data'][...]
        self.cmap_energies = (_E[1:]+_E[:-1])/2

        if self.Qpts_option == 'path':
            self.cmap_Qpt_distances = _cmap['x_data'][...]

        if self.Qpts_option == 'grid':
            _shape = list(self.Qpts_grid_shape)
            _shape.append(self.cmap_energies.size)

            self.cmap_structure_factors.shape = _shape

    # ----------------------------------------------------------------------------------------------

    def save_structure_factors_to_hdf5(self,hdf5_file='structure_factors.hdf5'):
        
        """
        write the structure factors to hdf5 file
        """

        with h5py.File(hdf5_file,'w') as db:

            db.create_dataset('raw_structure_factors',data=self.raw_structure_factors)
            db.create_dataset('raw_energies',data=self.raw_energies)
            db.create_dataset('Qpts',data=self.Qpts)

            if hasattr(self,'cmap_structure_factors'):
                db.create_dataset('cmap_structure_factors',data=self.cmap_structure_factors)
                db.create_dataset('cmap_energies',data=self.cmap_energies)

                if self.Qpts_option == 'path':
                    db.create_dataset('cmap_Qpt_distances',data=self.cmap_Qpt_distances)

                if self.Qpts_option == 'grid':
                    db.create_dataset('center',data=self.center)
                    db.create_dataset('proj',data=self.proj)
                    db.create_dataset('u_bins',data=self.u_bins)
                    db.create_dataset('v_bins',data=self.v_bins)
                    db.create_dataset('w_bins',data=self.w_bins)

    # ----------------------------------------------------------------------------------------------

    def plot_path_colormap(self,cmap='viridis',vmin=None,vmax=None,interpolation='none'):

        """
        plot the structure factors colormap
        """

        fig, ax = plt.subplots(figsize=(8,8))

        E = self.cmap_energies
        sqw = np.squeeze(self.cmap_structure_factors).T

        if hasattr(self,'cmap_Qpt_distances'):
            Q = self.cmap_Qpt_distances
        else:
            Q = np.linspace(0,1,sqw.shape[1])

        if vmax is None:
            vmax = sqw.max()*1e-14

        extent = [Q.min(),Q.max(),E.min(),E.max()]
        
        im = ax.imshow(sqw,origin='lower',aspect='auto',vmin=vmin,vmax=vmax,
                interpolation=interpolation,cmap=cmap,extent=extent)
        fig.colorbar(im,ax=ax,extend='both')

        ax.plot([Q.min(),Q.max()],[0,0],lw=2,ls=':',c='k',ms=0)

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1,labelsize='x-large')
        ax.tick_params(which='major',length=5)
        ax.tick_params(which='minor',length=2)

        ax.set_xlabel(r'|Q| [$\AA^{-1}$]',labelpad=4,fontsize='x-large')
        ax.set_ylabel('E [meV]',labelpad=4,fontsize='x-large')

        plt.show()
        plt.close()
        plt.clf()

    # ----------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

#    plot = True
    plot = False

    if plot:

        sqw = c_euphonic_sqw(cmap_file='structure_factors.hdf5')
        sqw.plot_colormap(vmax=1)

    else:

        sqw = c_euphonic_sqw()
    
        Qpt_vertices = [[1/2,1/2,1/2],
                        [0,0,0],
                        [1/2,1/2,0],
                        [0,0,0]]

        num_Qpt_steps = 101
        sqw.set_Qpts(Qpt_vertices,num_Qpt_steps)

        sqw.calculate_structure_factors(dw_grid=[24,24,24],temperature=300)
        sqw.get_colormap(E_min=-120,E_max=120,dE=0.1,E_width=2)

        sqw.save_structure_factors_to_hdf5()

        sqw.plot_colormap()
















