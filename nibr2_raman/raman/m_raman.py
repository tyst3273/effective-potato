
import numpy as np
import netCDF4 as nc
import h5py
import matplotlib.pyplot as plt
import scipy

ang2bohr = 1.88973
bohr2ang = 1/ang2bohr

ha2eV = 27.2114
ev2ha = 1/ha2eV

ev2meV = 1000
meV2eV = 1/ev2meV
meV2invcm = 8.065610

ha2meV = ha2eV * ev2meV

amu2me = 1822.89
me2amu = 1/amu2me

me2kg = 9.10938e-31
meV2thz = 0.2417991
thz2meV = 1/meV2thz


# --------------------------------------------------------------------------------------------------

class c_raman:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,phbst_file='run_PHBST.nc'):

        """
        ...
        """

        self.phbst_file = phbst_file

        self._parse_phbst_file()
        self._convert_displacements_to_eigenvectors()

        self._get_cartesian_positions()

    # ----------------------------------------------------------------------------------------------

    def _write_positions(self,filename,reduced_positions,template_text=None):

        """
        ...
        """

        _pos = reduced_positions

        with open(filename,'w') as f:

            if template_text is not None:
                f.write(template_text)
            
            _s = f'xred        {_pos[0,0]: 16.12f} {_pos[0,1]: 16.12f} {_pos[0,2]: 16.12f}\n'
            f.write(_s)
            for ii in range(1,self.num_atoms):
                _s = f'            {_pos[ii,0]: 16.12f} {_pos[ii,1]: 16.12f} {_pos[ii,2]: 16.12f}\n'
                f.write(_s)

    # ----------------------------------------------------------------------------------------------

    def get_raman_tensors(self,distance=0.02,template=None):

        if template is not None:
            with open(template,'r') as f:
                _template_text = f.read()
        else:
            _template_text = None

        self.raman_tensor = np.zeros((self.num_modes,3,3),dtype=float)

        for vv in range(self.num_modes):
            
            _f = self.freqs[0,vv]
            if _f < 1e-5:
                continue

            _filename = f'mode_{vv}_distance_{distance:.3f}.inp'
            _die_p = self._parse_dielectric_file()

            # _, _m = self.get_displaced_positions(vv,-distance)
            # _filename = f'mode_{vv}_distance_{-distance:.3f}.inp'
            # self._write_positions(_filename,_m,_template_text)

    # ----------------------------------------------------------------------------------------------

    def get_displacements_for_raman_calc(self,distance=0.02,template=None):

        if template is not None:
            with open(template,'r') as f:
                _template_text = f.read()
        else:
            _template_text = None

        for vv in range(self.num_modes):
            
            _f = self.freqs[0,vv]
            if _f < 1e-5:
                continue

            _, _p = self.get_displaced_positions(vv,distance)
            _filename = f'mode_{vv}_distance_{distance:.3f}.inp'
            self._write_positions(_filename,_p,_template_text)

            _, _m = self.get_displaced_positions(vv,-distance)
            _filename = f'mode_{vv}_distance_{-distance:.3f}.inp'
            self._write_positions(_filename,_m,_template_text)
            
    # ----------------------------------------------------------------------------------------------

    def get_displaced_positions(self,mode,distance=0.02):

        """
        ...
        """

        _disp = self.displacements[0,mode].real.reshape(self.num_atoms,3)
        _len = np.sqrt( np.sum(_disp**2,axis=1) ).max() 
        _disp = _disp * distance / _len

        self.cartesian_displaced_pos = self.cartesian_pos + _disp
        self.reduced_displaced_pos = np.zeros_like(self.cartesian_displaced_pos)

        _inv = self.inv_lattice_vectors
        for ii in range(self.num_atoms):
            self.reduced_displaced_pos[ii,:] = _inv @ self.cartesian_displaced_pos[ii,:]
        self.reduced_displaced_pos = self.reduced_displaced_pos.round(12)
        
        return self.cartesian_displaced_pos, self.reduced_displaced_pos
    
    # ----------------------------------------------------------------------------------------------

    def _get_cartesian_positions(self):

        """
        convert reduced atom positions to cartesian
        """
        
        _red = self.reduced_pos
        _lat = self.lattice_vectors

        self.cartesian_pos = np.zeros_like(_red)

        for ii in range(self.num_atoms):
            self.cartesian_pos[ii,:] = _lat @ _red[ii,:]

    # ----------------------------------------------------------------------------------------------

    def _parse_phbst_file(self):

        """
        parse the anaddb PHBST file

        from m_phonons.F90 in abinit:
            !!  Input data is in a.u, whereas the netcdf files saves data in eV for frequencies
            !!  and Angstrom for the displacements
            !!  The angular momentum is output in units of hbar
            
        """

        with nc.Dataset(self.phbst_file,'r') as ds:

            self.types = ds['atom_species'][...]
            self.num_atoms = self.types.size
            self.num_modes = self.num_atoms*3
            self.num_basis = self.num_atoms*3 
            
            self.masses = ds['atomic_mass_units'][...][self.types-1] #* amu2me

            self.reduced_pos = ds['reduced_atom_positions'][...] 

            # column vectors in bohr, i.e. (a, b, c) with a, b, c column vectors
            self.lattice_vectors = ds['primitive_vectors'][:,:].T 
            self.inv_lattice_vectors = np.linalg.inv(self.lattice_vectors)

            # shape = [num_qpts, num_modes, xyz] (its a 3d vector)
            self.phonon_angmom = ds['phangmom'][...] # units of hbar

            # shape = [num_qpts, num_modes, num_basis]. 
            self.displacements = ds['phdispl_cart'][...,0] + 1j*ds['phdispl_cart'][...,1] 
            self.displacements *= ang2bohr # now in Bohr
            self.freqs = ds['phfreqs'][...] * ev2ha # now in Ha
            self.num_qpts = self.freqs.shape[0]

    # ----------------------------------------------------------------------------------------------

    def _parse_dielectric_file(self,die_file):

        """
        parse the anaddb PHBST file

        from m_phonons.F90 in abinit:
            !!  Input data is in a.u, whereas the netcdf files saves data in eV for frequencies
            !!  and Angstrom for the displacements
            !!  The angular momentum is output in units of hbar
            
        """

        with nc.Dataset(self.die_file,'r') as ds:

            print(ds.variables)

            # self.types = ds['atom_species'][...]
            # self.num_atoms = self.types.size
            # self.num_modes = self.num_atoms*3
            # self.num_basis = self.num_atoms*3 
            
            # self.masses = ds['atomic_mass_units'][...][self.types-1] #* amu2me

            # self.reduced_pos = ds['reduced_atom_positions'][...] 

            # # column vectors in bohr, i.e. (a, b, c) with a, b, c column vectors
            # self.lattice_vectors = ds['primitive_vectors'][:,:].T 
            # self.inv_lattice_vectors = np.linalg.inv(self.lattice_vectors)

            # # shape = [num_qpts, num_modes, xyz] (its a 3d vector)
            # self.phonon_angmom = ds['phangmom'][...] # units of hbar

            # # shape = [num_qpts, num_modes, num_basis]. 
            # self.displacements = ds['phdispl_cart'][...,0] + 1j*ds['phdispl_cart'][...,1] 
            # self.displacements *= ang2bohr # now in Bohr
            # self.freqs = ds['phfreqs'][...] * ev2ha # now in Ha
            # self.num_qpts = self.freqs.shape[0

    # ----------------------------------------------------------------------------------------------

    def _convert_displacements_to_eigenvectors(self):
        
        """
        convert displacements to normalized eigenvectors
        """

        _sqrt_m = np.sqrt(self.masses)
        _mass_matrix = np.tile(_sqrt_m.reshape(self.num_atoms,1),reps=(1,3)).flatten()
        _mass_matrix = np.tile(_mass_matrix.reshape(1,self.num_basis),reps=(self.num_basis,1))

        # shape = [num_qpts, num_modes, num_basis]. units are angstrom
        self.eigenvectors = np.zeros( self.displacements.shape,dtype=complex)
        for qq in range(self.num_qpts):

            # convert disp. to eigs: eig_(q nu, a) = sqrt(m_a) disp_(q nu, a)
            self.eigenvectors[qq,...] = _mass_matrix * self.displacements[qq,...]

            # normalize
            for vv in range( self.num_modes):
                _eig =  self.eigenvectors[qq,vv,:]
                self.eigenvectors[qq,vv,:] /= np.sqrt(_eig.conj() @ _eig) 

        # # check normalization
        # _ovlp = np.zeros((self.num_modes,self.num_modes),dtype=float)
        # for qq in range(self.num_qpts):
        #     _ovlp[...] = 0.0

        #     for uu in range(self.num_modes):
        #         for vv in range(self.num_modes):
        #             _ovlp[uu,vv] = self.eigenvectors[qq,uu,:].conj() @  self.eigenvectors[qq,vv,:]

        #     print(_ovlp.round(6))

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    raman = c_raman('abinit/anaddb/run_PHBST.nc')
    # raman.get_displacements_for_raman_calc(template='template')

    raman.get_raman_tensors()








