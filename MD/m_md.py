import numpy as np


class c_md:

    def __init__(self,num_atoms=50,box_size=None,epsilon=1,sigma=1,masses=None):
        """
        class for NVT simulation of 1D LJ chain. setup a bunch of stuff we will need later

        NOTE: we are in units where kb=1
        """
        print('\n*** WARNING ***\npair potential probably needs a factor of 1/2 for total PE\n')

        self.num_atoms = int(num_atoms)

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

        # default is minimum energy for given LJ potential
        if box_size is None:
            box_size = 2**(1/6)*self.sigma*self.num_atoms
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

    def _setup_box(self):
        """
        equally spaced array of atoms filling box
        """
        self.pos = np.arange(self.num_atoms)/self.num_atoms*self.box_size

    def _get_neighbors(self):
        """
        loop over all atoms and get neighbor list for each
        """
        for ii in range(self.num_atoms):
            self.neighbor_vecs[ii,:] = self._do_minimum_image(self.pos-self.pos[ii])
        self.neighbor_dist[...] = np.abs(self.neighbor_vecs)

    def _do_minimum_image(self,pos):
        """
        apply minimum image to 
        """
        delta = (pos <= -self.box_size/2).astype(float)*self.box_size
        delta -= (pos >= self.box_size/2).astype(float)*self.box_size
        return pos+delta

    def _calculate_forces_and_potential(self):
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

        _dist *= mask

        return self.forces

    def run_nve(self,dt=1.0,num_steps=1000,write=True):
        """
        run NVE simulation using velocity verlet integration
        """
        
        # MD time step
        self.time_step = dt

        print('step, temp, ke, ...')

        if write:
            _fout = open('pos.xyz','w')

        for ii in range(num_steps):

            self._do_velocity_verlet()
            ke, T = self._calculate_kinetic_energy()

            msg = f'{ii+1:6} {T:6.3e} {ke:6.3e}'
            print(msg)
                
            if write:
                _fout.write(f'{self.num_atoms}\nstep {ii+1}\n')
                for jj in range(self.num_atoms):
                    _fout.write(f'C {self.pos[jj]: 9.6f}  0.0  0.0\n') 

    def _calculate_kinetic_energy(self):
        """
        calculate kinetic energy and temperature of the system 
        note: 1/2 m v**2 = d/2 * k_b * T where d is the dimension, here == 1.
        also note, k_b == 1.
        """
        self.ke = 1/2*np.sum(self.masses*self.vels**2)
        self.temperature = self.ke/2/self.num_atoms
        return self.ke, self.temperature

    def _do_velocity_verlet(self):
        """
        do a single velocity verlet step
        """
        _dt = self.time_step
        _m = self.masses

        # get forces and pot. and current positions
        _f = self._calculate_forces_and_potential()

        self.pos = self.pos + _dt*self.vels + _dt**2*_f/2/_m
        self.vels = self.vels + _dt*_f/2/_m

        # get forces and pot. and new positions
        _f = self._calculate_forces_and_potential()

        self.vels = self.vels + _dt*_f/2/_m



        


if __name__ == '__main__':

    md = c_md(10,None,epsilon=10,sigma=1/2**(1/6),masses=10)

    md.pos[0] = 0.2
    md.pos[1] += -0.05 
    md.pos[7] += 0.2

    md.run_nve(0.0005,20000,write=True)

