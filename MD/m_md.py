import numpy as np
import os
import h5py

# --------------------------------------------------------------------------------------------------

class c_md:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,num_atoms=50,box_size=None,epsilon=1,sigma=1,masses=None,atom_ids=None):
        """
        class for NVT simulation of 1D LJ chain. setup a bunch of stuff we will need later

        NOTE: we are in units where kb=1
        """

        self.num_atoms = int(num_atoms)
        self.step = 0

        self.traj_file = 'pos.xyz'
        if os.path.exists(self.traj_file):
            os.remove(self.traj_file)

        # LJ parameters: V(r) = 4 * eps * [ (sig/r)**12 - (sig/r)**6 ]
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)

        # masses of the atoms
        if masses is None:
            self.masses = np.ones(self.num_atoms)
        elif isinstance(masses,float):
            self.masses = np.ones(self.num_atoms)*masses
        else:
            self.masses = np.array(masses)

        # ids of the atoms (to map to primitive cell)
        if atom_ids is None:
            self.atom_ids = np.ones(self.num_atoms)
        else:
            self.atom_ids = np.array(atom_ids,dtype=int)

        # default is minimum energy for given LJ potential
        if box_size is None:
            box_size = self.num_atoms #2**(1/6)*self.sigma*self.num_atoms
        self.box_size = float(box_size)

        # neighbor vectors and distances
        self.neighbor_vecs = np.zeros((self.num_atoms,self.num_atoms))
        self.neighbor_dist = np.zeros((self.num_atoms,self.num_atoms))

        # forces and potential energy
        self.pair_forces = np.zeros((self.num_atoms,self.num_atoms))
        self.pair_potential = np.zeros((self.num_atoms,self.num_atoms))
        self.forces = np.zeros(self.num_atoms)
        self.potential = np.zeros(self.num_atoms)

        # velocities during MD integration
        self.vels = np.zeros(self.num_atoms)

        # create array of atoms
        self._setup_box()

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

    def _setup_box(self):
        """
        equally spaced array of atoms filling box
        """
        self.pos = np.arange(self.num_atoms)/self.num_atoms*self.box_size
    
    # ----------------------------------------------------------------------------------------------

    def _get_neighbors(self):
        """
        loop over all atoms and get neighbor list for each
        """
        for ii in range(self.num_atoms):
            self.neighbor_vecs[ii,:] = self._do_minimum_image(self.pos-self.pos[ii])
        self.neighbor_dist[...] = np.abs(self.neighbor_vecs)

    # ----------------------------------------------------------------------------------------------

    def _do_minimum_image(self,pos):
        """
        apply minimum image to 
        """
        delta = (pos <= -self.box_size/2).astype(float)*self.box_size
        delta -= (pos >= self.box_size/2).astype(float)*self.box_size
        return pos+delta

    # ----------------------------------------------------------------------------------------------

    def calculate_forces_and_potential(self):
        """
        calculate force on all atoms by summing over pairs 
        """

        # get neighbor lists
        self._get_neighbors()

        # neighbor distance
        _dist = self.neighbor_dist

        # LJ parameters
        _eps = self.epsilon
        _sig = self.sigma

        # need to mask the |r|==0 (i.e. i==j) terms
        mask = (_dist > 0.0).astype(float)
        _dist += (_dist == 0.0).astype(float) 

        # unit vectors
        _vecs = self.neighbor_vecs/_dist

        # calculate force and potential
        self.pair_forces[...] = -24*_eps*(2*(_sig/_dist)**12-(_sig/_dist)**6)/_dist*_vecs
        self.pair_potential[...] = 4*_eps*((_sig/_dist)**12-(_sig/_dist)**6)*mask
        self.forces[:] = self.pair_forces.sum(axis=1)
        self.potential[:] = self.pair_potential.sum(axis=1)
        self.pe = self.pair_potential.sum()/2

        _dist *= mask

        return self.forces

    # ----------------------------------------------------------------------------------------------

    def run_nve(self,dt=0.001,num_steps=1000):
        """
        run NVE simulation using velocity verlet integration
        """
        
        # MD time step
        self.time_step = dt

        print('# nve:  step,  temp,    ke,    pe,  etot')

        with open('pos.xyz','a') as _fout:

            for ii in range(num_steps):

                self._do_velocity_verlet()

                ke, T = self._calculate_ke_and_temperature()
                pe = self.pe
                etot = ke+pe

                msg = f'{self.step+1: 6} {T: .6e} {ke: .6e} {pe: .6e} {etot: .6e}'
                print(msg)
                
                _fout.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _fout.write(f'C {self.pos[jj]: 9.6f}  0.0  0.0\n') 
    
                self.step += 1

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

    def run_nvt_langevin(self,dt=0.001,num_steps=1000,T=1,damp=0.1):
        """
        run NVT simulation using velocity verlet integration and Langevin thermostat. note,
        damp has dims 1/time
        """

        # MD time step
        self.time_step = dt
        self.target_temperature = T
        self.damp = damp

        print('# nvt:  step,  temp,    ke,    pe,  etot')

        with open('pos.xyz','a') as _fout:

            for ii in range(num_steps):

                self._do_langevin_velocity_verlet()

                ke, T = self._calculate_ke_and_temperature()
                pe = self.pe
                etot = ke+pe

                msg = f'{self.step+1: 6} {T: .6e} {ke: .6e} {pe: .6e} {etot: .6e}'
                print(msg)

                _fout.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _fout.write(f'C {self.pos[jj]: 9.6f}  0.0  0.0\n')

                self.step += 1

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

    # ----------------------------------------------------------------------------------------------

    def run_nvt_nose_hoover(self,dt=0.001,num_steps=1000,T=1,Q=0.1,hdf5_file='nvt.hdf5'):
        """
        run NVT simulation using velocity verlet integration and nose-hoover thermostat. Q is the 
        the heat-bath "mass" (actually has dims M*L**2 in this algorithm?)
        """

        # MD time step
        self.time_step = dt
        self.target_temperature = T
        self.Q = Q
        
        # thermostat degree of freedom
        self.nose_dof = 0.0 

        print('# nvt:  step,  temp,    ke,    pe,  etot')

        with open('pos.xyz','a') as _fout, h5py.File(hdf5_file,'w') as db:

            db.create_dataset('masses',data=self.masses,dtype=float)
            db.create_dataset('steps',data=np.arange(num_steps),dtype=int)
            db.create_dataset('timestep',data=(self.time_step),dtype=float)
            db.create_dataset('target_temperature',data=(self.target_temperature),dtype=float)
            db.create_dataset('atom_ids',data=self.atom_ids,dtype=int)
            
            db.create_dataset('positions',shape=(num_steps,self.num_atoms),dtype=float)
            db.create_dataset('velocities',shape=(num_steps,self.num_atoms),dtype=float)
            db.create_dataset('etotal',shape=num_steps,dtype=float)
            db.create_dataset('temperature',shape=num_steps,dtype=float)

            for ii in range(num_steps):

                self._do_nose_hoover_velocity_verlet()

                ke, T = self._calculate_ke_and_temperature()
                pe = self.pe
                etot = ke+pe
                
                # write some data to hdf5 file
                db['positions'][ii,:] = self.pos[...]
                db['velocities'][ii,:] = self.vels[...]
                db['temperature'][ii] = T
                db['etotal'][ii] = etot

                msg = f'{self.step+1: 6} {T: .6e} {ke: .6e} {pe: .6e} {etot: .6e}'
                print(msg)

                _fout.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _fout.write(f'C {self.pos[jj]: 9.6f}  0.0  0.0\n')

                self.step += 1

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
        self.vels = self.vels + _dt*(_f/_m-_dof*self.vels)/2 # 

        # xi(t+dt/2)
        _dof = _dof + _dt/2/_Q*(_ke-_temp*(self.num_atoms+1)/2) 

        # now need ke at v(t+dt/2) 
        _ke, _ = self._calculate_ke_and_temperature()

        # forces at r(t+dt)
        _f = self.calculate_forces_and_potential()

        # xi(t+dt)
        _dof = _dof + _dt/2/_Q*(_ke-_temp*(self.num_atoms+1)/2) 

        # v(t+dt)
        self.vels = (self.vels + _dt*_f/2/_m) / (1 + _dt*_dof/2)

        # zero c.o.m.
        _v_cm = np.sum(self.masses*self.vels)/self.masses.sum()
        self.vels -= _v_cm/self.num_atoms

    # ----------------------------------------------------------------------------------------------
        
# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    calc_fc = True

    sigma = 1 #1/2**(1/6)
    epsilon = 1

    md = c_md(num_atoms=25,epsilon=1,masses=1)

    # ----------------------------------------------------------------------------------------------

    if calc_fc:

        # numerical force-constants
        pos = md.pos

        d = 1e-6

        pos[0] = d
        forces = md.calculate_forces_and_potential()
        fc = -forces/d

        pos[0] = -d
        forces = md.calculate_forces_and_potential()
        fc += forces/d

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

    # ----------------------------------------------------------------------------------------------

    #md.plot_potential()

    dt = 0.001
    temp = 0.1

    md.set_velocities(temp)
    #md.run_nve(dt=dt,num_steps=1000)

    #md.run_nvt_langevin(dt=dt,num_steps=5000,T=temp,damp=10)

    md.run_nvt_nose_hoover(dt=dt,num_steps=10000,T=temp,Q=0.001)

    md.run_nvt_nose_hoover(dt=dt,num_steps=50000,T=temp,Q=0.001)




