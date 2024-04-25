import numpy as np
import os
import h5py

# --------------------------------------------------------------------------------------------------

class c_md:

    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,pos,box_size,atom_types,masses,basis_index,epsilon=1,sigma=1,cutoff=None):
        """
        class for NVT simulation of 1D LJ chain. setup a bunch of stuff we will need later

        NOTE: we are in units where kb=1
        """

        # initialize parameters
        self.pos = np.array(pos,dtype=float)
        self.num_atoms = self.pos.size
        self.box_size = float(box_size)
        self.basis_index = np.array(basis_index,dtype=int)
        self.atom_types = np.array(atom_types,dtype=int)

        # create array of masses for each atom in simulation
        _masses = np.array(masses,dtype=float)
        _, _inv = np.unique(self.atom_types,return_inverse=True)
        self.masses = _masses[_inv]

        # LJ parameters: V(r) = 4 * eps * [ (sig/r)**12 - (sig/r)**6 ]
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)

        # cutoff to truncate forces etc
        if cutoff is None:
            self.cutoff = self.box_size
        else:
            self.cutoff = float(cutoff)

        # build neighbor lists once and for all
        self._build_neighbor_lists()
        
        # initializing these once and for all 

        # neighbor vectors and distances
        self.neighbor_vecs = np.zeros((self.num_atoms,self.num_neighbors))
        self.neighbor_dists = np.zeros((self.num_atoms,self.num_neighbors))

        # forces and potential energy
        self.pair_forces = np.zeros((self.num_atoms,self.num_neighbors))
        self.pair_potential = np.zeros((self.num_atoms,self.num_neighbors))
        self.forces = np.zeros(self.num_atoms)
        self.potential = np.zeros(self.num_atoms)

        # velocities during MD integration
        self.vels = np.zeros(self.num_atoms)

    # ----------------------------------------------------------------------------------------------

    def plot_potential(self,rmin=0.1,rmax=10,num_r=1001):
        """
        plot the LJ potential
        """
        import matplotlib.pyplot as plt
        r = np.linspace(rmin,rmax,num_r)
        _eps = self.epsilon
        _sig = self.sigma
        pot = 4*_eps*((_sig/r)**12-(_sig/r)**6)
        force = -24*_eps/r*(2*(_sig/r)**12-(_sig/r)**6)
        plt.plot(r,pot,lw=1,ms=0,c='b')
        plt.plot(r,force,lw=1,ms=0,c='r')
        plt.xlabel('r')
        plt.ylabel('V')
        plt.axis([rmin,rmax,-10,100])
        plt.show()

    # ----------------------------------------------------------------------------------------------

    def _build_neighbor_lists(self):
        """
        tiddies - claire

        get neighbor lists within cutoff to speedup force calculations later
        """

        _cutoff = self.cutoff

        _vecs = np.zeros((self.num_atoms,self.num_atoms),dtype=float)
        _dists = np.zeros((self.num_atoms,self.num_atoms),dtype=float)
        _inds = np.zeros((self.num_atoms,self.num_atoms),dtype=int)

        _num = 0
        for ii in range(self.num_atoms):
            _v = self._do_minimum_image(self.pos-self.pos[ii])
            _d = np.abs(_v)
            _i = np.argsort(_d)
            _dists[ii,:] = _d[_i]
            _vecs[ii,:] = _v[_i]
            _inds[ii,:] = _i

            _n = np.count_nonzero(_d <= _cutoff) 
            if _n > _num:
                _num = _n
        
        self.neighbor_lists = _inds[:,:_num]
        self.num_neighbors = _num

    # ----------------------------------------------------------------------------------------------

    def _get_neighbor_vecs_and_dists(self):
        """
        loop over all atoms and get neighbor list for each
        """
        for ii in range(self.num_atoms):
            _inds = self.neighbor_lists[ii,:] # atom ii's neighbors within cutoff
            self.neighbor_vecs[ii,:] = self._do_minimum_image(self.pos[_inds]-self.pos[ii])
        self.neighbor_dists[...] = np.abs(self.neighbor_vecs)

    # ----------------------------------------------------------------------------------------------

    def _do_minimum_image(self,pos):
        """
        apply minimum image to 
        """
        delta = (pos <= -self.box_size/2).astype(float)*self.box_size
        delta -= (pos >= self.box_size/2).astype(float)*self.box_size
        return pos+delta

    # ----------------------------------------------------------------------------------------------

    def _get_masks(self):
        """
        need to mask |r|==0 term., i.e. self-interaction
        """
        self._get_neighbor_vecs_and_dists()

        _dists = self.neighbor_dists

        # need to mask the |r|==0 (i.e. i==j) terms
        self.neighbor_dists_mask = (_dists > 0.0).astype(float)
        self.neighbor_dists_shift = (_dists == 0.0).astype(float)

    # ----------------------------------------------------------------------------------------------

    def calculate_forces_and_potential(self):
        """
        calculate force on all atoms by summing over pairs
        """

        # get neighbor lists
        self._get_neighbor_vecs_and_dists()

        # neighbor distance
        _dists = self.neighbor_dists 
        _dists[:,0] = 1.0 # mask the |r| == 0 term

        # LJ parameters
        _eps = self.epsilon
        _sig = self.sigma

        # unit vectors
        _unit = self.neighbor_vecs/_dists

        # calculate force and potential
        self.pair_forces[...] = -24*_eps*(2*(_sig/_dists)**12-(_sig/_dists)**6)/_dists*_unit
        self.pair_potential[...] = 4*_eps*((_sig/_dists)**12-(_sig/_dists)**6)

        # mask the |r| == 0.0 terms
        self.pair_forces[:,0] = 0.0 
        self.pair_potential[:,0] = 0.0

        # sum over pairs
        self.forces[:] = self.pair_forces.sum(axis=1)
        self.potential[:] = self.pair_potential.sum(axis=1)
        self.pe = self.pair_potential.sum()/2

        _dists[:,0] = 0.0 # reset the |r|==0 term to 0

        return self.forces

    # ----------------------------------------------------------------------------------------------

    def run_nve(self,dt=0.001,num_steps=1000,xyz_file='nve.xyz'):
        """
        run NVE simulation using velocity verlet integration
        """
        
        # MD time step
        self.time_step = dt
        
        # calculate initial PE to write to log file ...
        self.calculate_forces_and_potential()

        print('# nve:  step,  temp,    ke,    pe,  etot')
        
        with open(xyz_file,'w') as _fout:

            for ii in range(num_steps):

                ke, T = self._calculate_ke_and_temperature()
                pe = self.pe
                etot = ke+pe

                msg = f'{ii: 6} {T: .6e} {ke: .6e} {pe: .6e} {etot: .6e}'
                print(msg)
                
                _fout.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _fout.write(f'C {self.pos[jj]: 9.6f}  0.0  0.0\n') 

                self._do_velocity_verlet()
    
    # ----------------------------------------------------------------------------------------------

    def _calculate_ke_and_temperature(self):
        """
        calculate kinetic energy and temperature of the system 
        note: 1/2 m v**2 = d/2 * k_b * T where d is the dimension, here == 1.
        also note, k_b == 1.
        """
        self.ke = 1/2*np.sum(self.masses*self.vels**2)
        self.temperature = 2*self.ke/self.num_atoms
        return self.ke, self.temperature

    # ----------------------------------------------------------------------------------------------

    def _do_velocity_verlet(self):
        """
        do a single velocity verlet step
        """
        _dt = self.time_step
        _m = self.masses

        # get forces and pot. and current positions
        _f = self.calculate_forces_and_potential()

        self.pos = self.pos + _dt*self.vels + _dt**2*_f/2/_m
        self.vels = self.vels + _dt*_f/2/_m

        # get forces and pot. and new positions
        _f = self.calculate_forces_and_potential()

        self.vels = self.vels + _dt*_f/2/_m

        # zero com ...
        _v_cm = np.sum(self.masses*self.vels)/self.masses.sum()
        self.vels -= _v_cm/self.num_atoms

    # ----------------------------------------------------------------------------------------------

    def set_velocities(self,T=1,zero_drift=True):
        """
        set initial velocites of the system using box-muller sampling 
        """

        _num_sample = self.num_atoms

        _u1 = np.random.uniform(size=_num_sample)
        _u2 = np.random.uniform(size=_num_sample)

        self.vels[...] = np.sqrt(T/self.masses)*np.sqrt(-2*np.log(_u2))*np.cos(2*np.pi*_u1)

        # bonus numbers with the same dist. and vels above. we don't need it
        #_v = np.sqrt(-2*np.log(_u1))*np.sin(2*np.pi*_u2)

        if zero_drift:
            _v_cm = np.sum(self.masses*self.vels)/self.masses.sum()
            self.vels -= _v_cm/self.num_atoms

        #_, T = self._calculate_ke_and_temperature()
        #return T

    # ----------------------------------------------------------------------------------------------

    def run_nvt_langevin(self,dt=0.001,num_steps=1000,T=1,damp=0.1,xyz_file='langevin.xyz'):
        """
        run NVT simulation using velocity verlet integration and Langevin thermostat. note,
        damp has dims 1/time
        """

        msg = '\n*** WARNING ***\n'
        msg += 'i found a dissertation (in dir.) that compared transport and other\n'
        msg += 'things w/ langevin vs. nose-hooever. transport and phonon-dos are\n'
        msg += 'independent of nose-hoover damping, while they depend STRONGLY on\n'
        msg += 'langevin damping. dont use langevin for phonons!\n'
        print(msg)

        # MD time step
        self.time_step = dt
        self.target_temperature = T
        self.damp = damp

        # calculate initial PE to write to log file ...
        self.calculate_forces_and_potential()

        print('# nvt:  step,  temp,    ke,    pe,  etot')

        with open(xyz_file,'w') as _fout:

            for ii in range(num_steps):

                ke, T = self._calculate_ke_and_temperature()
                pe = self.pe
                etot = ke+pe

                msg = f'{ii: 6} {T: .6e} {ke: .6e} {pe: .6e} {etot: .6e}'
                print(msg)

                _fout.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _fout.write(f'C {self.pos[jj]: 9.6f}  0.0  0.0\n')

                self._do_langevin_velocity_verlet()

    # ----------------------------------------------------------------------------------------------

    def _do_langevin_velocity_verlet(self):
        """
        do a single langevin velocity verlet step according to ref 
            https://doi.org/10.1103/PhysRevE.75.056707
        """
        _dt = self.time_step
        _m = self.masses
        _damp = self.damp
        _temp = self.target_temperature
        _num_sample = self.num_atoms

        _c1 = np.exp(-_damp*_dt/2)
        _c2 = np.sqrt((1-_c1**2)*_m*_temp)
        _eta1 = np.random.normal(size=_num_sample)
        _eta2 = np.random.normal(size=_num_sample)

        # get forces and pot. and current positions
        _f1 = np.copy(self.calculate_forces_and_potential())

        self.vels = _c1*self.vels + _c2*_eta1/_m
        self.pos = self.pos + self.vels*_dt + _f1*(_dt)**2/2/_m

        # get forces and pot. and new positions
        _f2 = np.copy(self.calculate_forces_and_potential())

        self.vels = self.vels + (_f1+_f2)*_dt/2/_m
        self.vels = _c1*self.vels + _c2*_eta2/_m

        # zero com ...
        _v_cm = np.sum(self.masses*self.vels)/self.masses.sum()
        self.vels -= _v_cm/self.num_atoms

    # ----------------------------------------------------------------------------------------------

    def run_nvt_nose_hoover(self,dt=0.001,num_steps=1000,T=1,damp=0.1,hdf5_file=None,xyz_file=None,
            log_stride=100):
        """
        run NVT simulation using velocity verlet integration and nose-hoover thermostat. 
        the intergation scheme is based on the algo in 'nvt.pdf', but i changed (3N+1) => (3N) 
        in the velocitiy verlet step. it was giving the wrong target T.
        
        damp is related to Q in 'nvt.pdf' by damp**2 = Q/(3NkT)
            -- in 1d, it is damp**2 = Q/(NKT)

        Q is the the heat-bath "mass" (actually has dims M*L**2 in this algorithm?)
        """

        # MD time step
        self.time_step = dt
        self.target_temperature = T

        self.Q = damp**2*self.num_atoms*self.target_temperature
        
        # thermostat degree of freedom
        self.nose_dof = 0.0 

        # calculate initial PE to write to log file ...
        self.calculate_forces_and_potential()
        
        # whether or not to write hdf5
        write_hdf5 = False
        if hdf5_file is not None:
            write_hdf5 = True
            _hdf5 = h5py.File(hdf5_file,'w')
            _hdf5.create_dataset('masses',data=self.masses,dtype=float)
            _hdf5.create_dataset('steps',data=np.arange(num_steps),dtype=int)
            _hdf5.create_dataset('timestep',data=(self.time_step),dtype=float)
            _hdf5.create_dataset('target_temperature',data=(self.target_temperature),dtype=float)
            _hdf5.create_dataset('atom_types',data=self.atom_types,dtype=int)
            _hdf5.create_dataset('basis_index',data=self.basis_index,dtype=int)
            _hdf5.create_dataset('box_size',data=self.box_size,dtype=float)
            _hdf5.create_dataset('positions',shape=(num_steps,self.num_atoms),dtype=float)
            _hdf5.create_dataset('velocities',shape=(num_steps,self.num_atoms),dtype=float)
            _hdf5.create_dataset('etotal',shape=num_steps,dtype=float)
            _hdf5.create_dataset('temperature',shape=num_steps,dtype=float)
        
        # whether of not to write xyz file
        write_xyz = False
        if xyz_file is not None:
            write_xyz = True
            _xyz = open(xyz_file,'w')

        # header of log to 
        print('# nvt:  step,  temp,    ke,    pe,  etot')

        for ii in range(num_steps):

            ke, T = self._calculate_ke_and_temperature()
            pe = self.pe
            etot = ke+pe
            
            if write_hdf5:
                _hdf5['positions'][ii,:] = self.pos[...]
                _hdf5['velocities'][ii,:] = self.vels[...]
                _hdf5['temperature'][ii] = T
                _hdf5['etotal'][ii] = etot

            if write_xyz:
                _xyz.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _xyz.write(f'C {self.pos[jj]: 15.9f}  0.0  0.0\n')

            if ii % log_stride == 0: 
                msg = f'{ii: 6} {T: .6e} {ke: .6e} {pe: .6e} {etot: .6e}'
                print(msg)

            # update positions, velocities, forces, etc
            self._do_nose_hoover_velocity_verlet()

    # ----------------------------------------------------------------------------------------------

    def _do_nose_hoover_velocity_verlet(self):
        """
        do a single nose-hoover velocity verlet step
        """
        _dt = self.time_step
        _m = self.masses
        _Q = self.Q
        _temp = self.target_temperature
        _dof = self.nose_dof

        # need ke at v(t) 
        _ke, _ = self._calculate_ke_and_temperature()

        # forces at r(t)
        _f = self.calculate_forces_and_potential()

        # r(t+dt)
        self.pos = self.pos + _dt*self.vels + _dt**2*(_f/_m-_dof*self.vels)/2

        # v(t+dt/2)
        self.vels = self.vels + _dt*(_f/_m-_dof*self.vels)/2 

        # xi(t+dt/2)
        #_dof = _dof + _dt/2/_Q*(_ke-_temp*(self.num_atoms+1)/2) 
        _dof = _dof + _dt/2/_Q*(_ke-_temp*self.num_atoms/2)

        # now need ke at v(t+dt/2) 
        _ke, _ = self._calculate_ke_and_temperature()

        # forces at r(t+dt)
        _f = self.calculate_forces_and_potential()

        # xi(t+dt)
        #_dof = _dof + _dt/2/_Q*(_ke-_temp*(self.num_atoms+1)/2) 
        _dof = _dof + _dt/2/_Q*(_ke-_temp*self.num_atoms/2)

        # v(t+dt)
        self.vels = (self.vels + _dt*_f/2/_m) / (1 + _dt*_dof/2)

        # zero com ...
        _v_cm = np.sum(self.masses*self.vels)/self.masses.sum()
        self.vels -= _v_cm/self.num_atoms

    # ----------------------------------------------------------------------------------------------

    def read_restart(self,restart_file):
        """
        read velocities and positions from a previous hdf5 file to restart at a previous 
        configuration
        """

        with h5py.File(restart_file,'r') as db:
            _vels = db['velocities'][-1,:]
            _pos = db['positions'][-1,:]
            _box_size = db['box_size'][...]

        _num_atoms = _vels.size
        if _num_atoms != self.num_atoms:
            msg = '\nnumber of atoms in restart file doesnt match simulation!\n'
            exit(msg)
        if _box_size != self.box_size:
            msg = '\nbox size in restart file doesnt match simulation!\n'
            exit(msg)

        self.vels[...] = _vels
        self.pos[...] = _pos

    # ----------------------------------------------------------------------------------------------
        
# --------------------------------------------------------------------------------------------------

def calc_fc(md,d_pos=1e-6):
    """
    calculate and print force constants for defined model
    """
    # numerical force-constants
    pos = md.pos

    pos[0] = d_pos
    forces = md.calculate_forces_and_potential()
    fc = -forces/d_pos

    pos[0] = -d_pos
    forces = md.calculate_forces_and_potential()
    fc += forces/d_pos

    fc /= 2
    pos[0] = 0.0

    inds = np.argsort(pos)
    pos = pos[inds]
    fc = fc[inds]

    msg = ''
    for ii in range(md.num_atoms):
        msg += f'{pos[ii]: 3.2f} {fc[ii]: 9.6f}\n'
    print(msg)

    exit()

# --------------------------------------------------------------------------------------------------

def check_force_cutoff(md,d_pos=0.0,tol=1e-9):
    """
    shift 0th atom by d_pos and calculate forces and find distance where forces are >= tol
    """

    md.pos[0] = d_pos
    md.calculate_forces_and_potential()
    forces = md.pair_forces[0,:]
    vecs = md.neighbor_vecs[0,:]
    dist = md.neighbor_dists[0,:]

    msg = 'vec, dist, force\n'
    for ii in range(md.num_neighbors):
        msg += f'{vecs[ii]: 9.3f} {dist[ii]: 9.3f} {forces[ii]: 16.9f}\n'
    print(msg)

    exit()

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # define simulation
    num_atoms = 100
    pos = np.arange(num_atoms)
    box_size = num_atoms
    atom_types = np.ones(num_atoms)
    basis_index = np.ones(num_atoms)
    masses = [1]

    md = c_md(pos,box_size,atom_types,masses,basis_index,epsilon=1,sigma=1,cutoff=30)

    #calc_fc(md)
    #check_force_cutoff(md)

    dt = 0.001
    damp = 0.01
    temp = 0.1

    md.set_velocities(temp)
    #md.read_restart('restart.hdf5')

    #md.run_nve(dt=dt,num_steps=100000)

    md.run_nvt_nose_hoover(dt=dt,num_steps=100000,T=temp,damp=damp,hdf5_file='nvt_equil.hdf5')
    md.run_nvt_nose_hoover(dt=dt,num_steps=100000,T=temp,damp=damp,hdf5_file='nvt_run.hdf5')



