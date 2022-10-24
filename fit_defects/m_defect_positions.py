

import numpy as np
import matplotlib.pyplot as plt



class c_defect_positions:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):

        """
        create correlated set of defect positions
        """

        self.reps = np.array([20,20,20])
        self.num_sites = np.prod(self.reps)
        self.box_vectors = np.array([[5.0, 0.0, 0.0],    
                                     [0.0, 5.0, 0.0],
                                     [0.0, 0.0, 5.0]])

        self.rmc_tol = 1e-5
        self.rmc_sigma = 0.0001

        self._make_box()

    # ----------------------------------------------------------------------------------------------

    def _make_box(self):

        """
        arrays etc. that we need
        """

        self.vector_shape = [self.reps[0],self.reps[1],self.reps[2],3]
        self.scalar_shape = [self.reps[0],self.reps[1],self.reps[2]]
        self.flat_shape = [self.num_sites,3]

        # integer grid of indices
        _x = np.arange(self.reps[0])
        _y = np.arange(self.reps[1])
        _z = np.arange(self.reps[2])
        self.grid = np.array(np.meshgrid(_x,_y,_z,indexing='ij'))
        self.grid = np.moveaxis(self.grid,0,3)

        # cartesian grid
        _x = _x/self.reps[0]
        _y = _y/self.reps[1]
        _z = _z/self.reps[2]
        self.r_grid = np.array(np.meshgrid(_x,_y,_z,indexing='ij'))
        self.r_grid = np.moveaxis(self.r_grid,0,3)

        # convert reduced to cartesian coords
        self.r_grid.shape = self.flat_shape
        for ii in range(self.num_sites):
            self.r_grid[ii,:] = self.box_vectors[0,:]*self.r_grid[ii,0]+ \
                                self.box_vectors[1,:]*self.r_grid[ii,1]+ \
                                self.box_vectors[2,:]*self.r_grid[ii,2]
        self.r_grid.shape = self.vector_shape
          
        self.occ = np.zeros(self.scalar_shape)
        self.site_inds = np.arange(self.num_sites)

    # ----------------------------------------------------------------------------------------------

    def fit_radial_correlation(self,num_defects,r,corr_in,weights=None):

        """
        fit input radial correlation function using RMC
        """

        # weights for input correlation
        _nr = r.size
        if weights is None:
            self.weights = np.ones(_nr)
        else:
            self.weights = np.array(weights)

        _sites = np.copy(self.site_inds)
        np.random.shuffle(_sites)
        _inds = _sites[:num_defects]

        # make initial guess for occupations
        _coords = self.grid.reshape(self.flat_shape)[_inds,:]
        self.set_occupations(_coords)

        # calculate initial corr func.
        self.get_real_space_corr()
        self.get_on_radial_grid()

        # calculate initial error
        radial_corr = np.interp(r,self.radial_bins,self.radial_corr,right=0)
        radial_corr = radial_corr/radial_corr.max()*corr_in.max() # DEV: norm. to match input
        chi_o = np.sum((corr_in-radial_corr)**2)/num_defects

        # now go and interate to minimize err wrt input corr func.
        converged = False
        max_iter = 100000
        count = 0
        while True:

            accept = False
            
            # break if too many iterations
            if count > max_iter:
                print(f'\nthe rmc loop didnt converge after {max_iter} iterations!\n')
                exit()

            # move
            _new_coords = np.copy(_coords)
            _new_coords = self.move_defects(_new_coords)
            self.set_occupations(_new_coords)

            # calculate corr func.
            self.get_real_space_corr()
            self.get_on_radial_grid()

            # calculate error
            radial_corr = np.interp(r,self.radial_bins,self.radial_corr,right=0)
            radial_corr = radial_corr/radial_corr.max()*corr_in.max() # DEV: norm. to match input

            if False:
                plt.plot(r,radial_corr,c='r')
                plt.plot(r,corr_in,c='b')
                plt.show()
                plt.clf()

            chi_n = np.sum((corr_in-radial_corr)**2)/num_defects

            # otherwise, keep going
            delta_chi = chi_n-chi_o

            # check if converged
            #if np.abs(delta_chi) < self.rmc_tol:
            if chi_n < self.rmc_tol:
                converged = True
                print('\nthe rmc loop converged!\n')
                break

            # accept move 
            if delta_chi < 0:
                accept = True

            # otherwise, maaaybe accept move
            if delta_chi >= 0:
                p = np.exp(-delta_chi/self.rmc_sigma)
                #print(p)
                if p > np.random.uniform():
                    accept = True

            if accept:
                print(f'\ncount: {count}')
                print(f'chi_n: {chi_n}')
                print(f'delta_chi: {delta_chi}')
                _coords = np.copy(_new_coords)
                chi_o = chi_n
                count += 1


        _t = np.zeros((self.num_sites,4))
        _t[:,0] = self.occ.flatten()
        _t[:,1:] = self.r_grid.reshape(self.flat_shape)
        np.savetxt('t',_t,fmt='%g %9.6f %9.6f %9.6f')

        plt.plot(r,radial_corr,c='r')
        plt.plot(r,corr_in,c='b')
        plt.show()
        plt.clf()
                    
    # ----------------------------------------------------------------------------------------------

    def move_defects(self,_coords,dx=8):

        """
        randomly displace a random number of defects to neighboring unitcells
        """

        #_d = np.array([-1,0,1])
        _d = np.arange(-dx,dx+1)

        _i = np.arange(_coords.shape[0])
        np.random.shuffle(_i)
        _n = _i[0]

        np.random.shuffle(_i)
        _i = _i[:_n]

        for ii in range(_coords.shape[0]):
            for jj in range(3):
                np.random.shuffle(_d)   
                _coords[ii,jj] += _d[0]
                if _coords[ii,jj] >= self.reps[jj]:
                    _coords[ii,jj] += -self.reps[jj]
                if _coords[ii,jj] < 0:
                    _coords[ii,jj] += self.reps[jj]

        return _coords

    # ----------------------------------------------------------------------------------------------

    def get_real_space_corr(self):

        """
        calculate real space correlation function
        """

        _c = np.fft.fftn(self.occ)
        self.corr = np.real(np.fft.ifftn(np.abs(_c)**2))

    # ----------------------------------------------------------------------------------------------

    def get_on_radial_grid(self):

        """
        put real space corr. func on radial grid
        """

        _r = np.sqrt(np.sum(self.r_grid**2,axis=3))
        _r = _r.flatten()
        _c = self.corr.flatten()

        _bins = np.unique(_r)
        _counts, _ = np.histogram(_r,bins=_bins)
        _hist, _ = np.histogram(_r,bins=_bins,weights=_c) # histogram sig 

        self.radial_corr = _hist/_counts

        # radial bins start from 0
        self.radial_bins = _bins[0:-1]

    # ----------------------------------------------------------------------------------------------

    def set_occupations(self,_coords):

        """
        got and set the occupations to 1 for the specified coords
        """

        self.occ[...] = 0
        for ii in range(_coords.shape[0]):
            self.occ[_coords[ii,0],_coords[ii,1],_coords[ii,2]] = 1

    # ----------------------------------------------------------------------------------------------





# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    defect_pos = c_defect_positions()


    nr = 100
    xi = 0.5
    r = np.linspace(0,5,nr)
    f = np.exp(-np.abs(r)/xi)

    num_defects = 100
    defect_pos.fit_radial_correlation(num_defects,r,f)
        














