
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
            self.box_size = db['box_size'][...]
            self.temperature = db['temperature'][...]
    
        self._crop_trajectory(skip,trim,stride)
        #self._get_freqs()

        self.mean_pos = self.pos.mean(axis=0)
        _min = self.mean_pos.min()
        self.mean_pos += -_min
        self.pos += -_min

        self.displacement = self.pos-self.mean_pos

        self.temperature = self.temperature.mean()

        self.num_atoms = self.pos.shape[1]

        self._get_types()

    # ----------------------------------------------------------------------------------------------

    def _get_types(self):

        self.unique_types, self.type_counts = np.unique(self.atom_types,return_counts=True)
        self.num_types = self.unique_types.size

        _n = self.num_types
        self.num_pairs = int(_n*(_n+1)/2)
        self.type_pairs = np.zeros((self.num_pairs,2),dtype=int)

        _s = 0
        for ii in range(_n):
            for jj in range(ii,_n):
                self.type_pairs[_s,:] = [ii,jj]
                _s += 1

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
            self.num_qpts = self.num_atoms+1
        else:
            self.num_qpts = num_qpts
        
        dq = 1/self.num_qpts
        self.qpts = np.linspace(-0.5,0.5,self.num_qpts)
        self.qpts_cart = self.qpts*2*np.pi/a

        print('qpts:',self.qpts.round(3))

    # ----------------------------------------------------------------------------------------------

    def calc_dynmat(self):
        
        self.dynmat = np.zeros((self.num_qpts,self.num_types,self.num_types),dtype=complex)

        for ii in range(self.num_qpts):
            
            qpt = self.qpts_cart[ii]
            self._loop_over_atom_pairs(qpt)

            _q = np.ones(self.num_steps)*self.qpts_cart[ii]

    # ----------------------------------------------------------------------------------------------

    def _loop_over_atoms(self,qpt):
            
        qpt = np.ones(self.num_steps)*qpt

        _n = self.num_types
        for ii in range(_n):
            for jj in range(ii,_n):
                
                pass

    # ----------------------------------------------------------------------------------------------

if __name__ == '__main__':

    calc = c_calc_fc('keep.hdf5',stride=100)

    calc.get_qpts()
    calc.calc_dynmat()

