
import matplotlib.pyplot as plt
import numpy as np
import h5py

class c_calc_fc:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,hdf5_file,skip=0,trim=0,stride=1):

        with h5py.File(hdf5_file,'r') as db:
            self.pos = db['positions'][...]
            self.vels = db['velocities'][...]
            self.time_step = db['timestep'][...]
            self.steps = db['steps'][...]
            self.masses = db['masses'][...]
            self.atom_types = db['atom_types'][...]
            self.basis_index = db['basis_index'][...]
            self.box_size = db['box_size'][...]
            self.temperature = db['temperature'][...]

        self._crop_trajectory(skip,trim,stride)
        self.num_atoms = self.pos.shape[1]

        #self._get_freqs()

        self.mean_pos = self.pos.mean(axis=0)
        _min = self.mean_pos.min()
        self.mean_pos += -_min
        self.pos += -_min
        self.displacements = self.pos-self.mean_pos

        # reshape for vectorized fourier transforms later
        self.mean_pos = np.tile(self.mean_pos.reshape(1,self.num_atoms),reps=(self.num_steps,1))

        self.temperature = self.temperature.mean()

        self._get_types()

    # ----------------------------------------------------------------------------------------------

    def _get_types(self):

        self.unique_basis, self.basis_counts = np.unique(self.basis_index,return_counts=True)
        self.num_basis = self.unique_basis.size

        # should be ordered, i.e. same basis in each unitcell.
        self.num_unitcells = self.basis_counts[0]

        _n = self.num_basis
        self.num_pairs = int(_n*(_n+1)/2)
        self.basis_pairs = np.zeros((self.num_pairs,2),dtype=int)

        _s = 0
        for ii in range(_n):
            for jj in range(ii,_n):
                self.basis_pairs[_s,:] = [ii,jj]
                _s += 1

        self.inds_per_basis = [[] for _ in range(_n)]
        for ii in range(_n):
            ind = self.unique_basis[ii]
            self.inds_per_basis[ii] = np.flatnonzero(self.basis_index == ind)

    # ----------------------------------------------------------------------------------------------

    def _get_freqs(self):

        _dt = (self.steps[1]-self.steps[0])*self.time_step

        self.num_freqs = self.num_steps

        self.freqs = np.fft.fftfreq(self.num_freqs,_dt)
        self.freq_step = self.freqs[1]-self.freqs[0]
        self.freq_max = self.freqs.max()

        print('time-step:',_dt)
        print('freq-step:',self.freq_step)
        print('freq-max:',self.freq_max)
        print('freqs:',self.freqs)

    # ----------------------------------------------------------------------------------------------

    def _crop_trajectory(self,skip=0,trim=0,stride=1):

        _num = self.steps.size
        
        self.pos = self.pos[skip:_num-trim:stride,:]
        self.vels = self.vels[skip:_num-trim:stride,:]

        self.steps = self.steps[skip:_num-trim:stride]
        self.steps += -self.steps[0]

        self.num_steps = self.steps.size

    # ----------------------------------------------------------------------------------------------

    def get_qpts(self,num_qpts=None,a=1.0):

        if num_qpts is None:
            self.num_qpts = self.num_unitcells+1
        else:
            self.num_qpts = num_qpts
        
        dq = 1/self.num_qpts
        self.qpts = np.linspace(-0.5,0.5,self.num_qpts)
        self.qpts_cart = self.qpts*2*np.pi/a

        print('qpts:',self.qpts.round(3))

    # ----------------------------------------------------------------------------------------------

    def calc_dynmat(self):
        
        self.dynmat = np.zeros((self.num_qpts,self.num_basis,self.num_basis),dtype=complex)

        _num_basis = self.num_basis

        for qq in range(self.num_qpts):
            
            print(f'now on qpt {qq+1} out of {self.num_qpts}')

            self.this_qpt = np.ones((self.num_steps,self.num_unitcells))*self.qpts_cart[qq]

            for ii in range(_num_basis):
                u_q_ii = self._get_displacement_ft(ii)

                for jj in range(ii,_num_basis):
                    u_q_jj = self._get_displacement_ft(jj)

                    self.dynmat[qq,ii,jj] = 2/self.temperature/np.mean(u_q_jj.conj()*u_q_ii)

        freq = np.sqrt(self.dynmat[:,0,0])
        plt.plot(np.linspace(0,1,self.num_qpts),freq)
        plt.axis([0,1,0,10000])
        plt.show()

    # ----------------------------------------------------------------------------------------------

    def _get_displacement_ft(self,basis_ind):

        # this basis atom int he whole supercell
        _inds = self.inds_per_basis[basis_ind]
        _num_unitcells = self.num_unitcells

        # get mean pos and displacement 
        _pos = self.mean_pos[:,_inds]
        _disp = self.displacements[:,_inds]

        # vectorized FT
        _qpt = self.this_qpt
        _exp_iqr = np.exp(-1j*_qpt*_pos)
        u_q = np.sum(_disp*_exp_iqr,axis=1)/np.sqrt(_num_unitcells)

        return u_q

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    calc = c_calc_fc('keep.hdf5',stride=100)

    calc.get_qpts(num_qpts=101)
    calc.calc_dynmat()











