
import numpy as np
import h5py

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,y=0.1,z=0.0,x_lo=None,x_hi=5,num_x=10001):

        """
        dot U = 0 = v^2 n / x + y^4 - x^4 
        dot n = 0 = e^(-1/x) - n

        v is dimensionless voltage, x is dimensionless sample temperature, y is 
            dimensionless bath temperature. n is defect concentration.

        v and y are "knobs" that we turn in the lab. x and n are variables we solve for. 
        """

        self.y = y
        self.z = z

        if x_lo is None:
            x_lo = y
        self.x = np.linspace(x_lo,x_hi,num_x)

    # ----------------------------------------------------------------------------------------------

    def solve_constant_v(self,v):

        """
        ...
        """

        self.v = v 
        self.v_sq = v**2
        print(f'\nconstant v: {v:6.4f}')

        _x = self.x
        _n = self._calc_n_constant_v(_x)

        x_0 = self._solve_dot_U_steady_state_constant_v(_n)
        n_0 = self._calc_n_constant_v(x_0)

        msg = '\n*** RESULTS ***'
        msg += f'\n\n% v: {self.v:9.6f}'
        msg += f'\n% y: {self.y:9.6f}'
        msg += f'\n% z: {self.z:9.6f}' 
        print(msg)

        print('\n% n:',n_0)
        print('% x:',x_0)

        return n_0, x_0

    # ----------------------------------------------------------------------------------------------

    def solve_constant_j(self,j,n_lo_guess=0.001,n_hi_guess=0.999,max_iter=200,n_tol=1e-6,
                               alpha=0.4):

        """
        we look for two solutions for n. we have to solve self-consistently, so how we find 
            different solutions is we have to make a guess close the different solutions. 
            we just pick n->0 and n->1 and solve for each. if they converge to the same solution, 
            there is only one. if they are different, there is multistability!
        """

        self.j = j 
        self.j_sq = j**2
        print(f'\nconstant j: {j:6.4f}')

        # if n_lo_guess < self.j * self.y * self.z:
        #     n_lo_guess = self.j * self.y * self.z+n_tol

        # low n guess
        print(f'\nsolving for low n guess: n_in={n_lo_guess:6.4f}')
        n_lo, x_lo, res_lo = \
            self._solve_self_consistent_constant_j(n_lo_guess,max_iter,n_tol,alpha)
        cond_lo = n_lo / x_lo
        v_lo = j / cond_lo

        # hi n guess
        print(f'\nsolving for high n guess: n_in={n_hi_guess:6.4f}')
        n_hi, x_hi, res_hi = \
            self._solve_self_consistent_constant_j(n_hi_guess,max_iter,n_tol,alpha)
        cond_hi = n_lo / x_lo
        v_hi = j / cond_hi

        n_diff = n_hi-n_lo
        x_diff = x_hi-x_lo

        msg = '\n*** RESULTS ***'
        msg += f'\n\n% j: {self.j:9.6f}'
        msg += f'\n% y: {self.y:9.6f}'
        msg += f'\n% z: {self.z:9.6f}' 

        msg += f'\n\n% n_lo: {n_lo:9.6e}'
        msg += f'\n% x_lo: {x_lo:9.6e}'
        msg += f'\n% cond_lo: {cond_lo:9.6e}'
        msg += f'\n% v_lo: {v_lo:9.6e}'
        msg += f'\n% residual_lo: {res_lo:9.6e}'

        msg += f'\n\n% n_hi: {n_hi:9.6e}'
        msg += f'\n% x_hi: {x_hi:9.6e}'
        msg += f'\n% cond_hi: {cond_lo:9.6e}'
        msg += f'\n% v_hi: {v_hi:9.6e}'
        msg += f'\n% residual_hi: {res_hi:9.6e}'

        msg += f'\n\n% n_diff: {n_diff:6.4f}'
        msg += f'\n% x_diff: {x_diff:6.4f}'
        print(msg)

        return n_lo, x_lo, n_hi, x_hi

    # ----------------------------------------------------------------------------------------------

    def _solve_self_consistent_constant_j(self,n_in,max_iter,n_tol,alpha):
        
        """
        solve for n self consistently: make a guess for n_in and calculate x from dot U = 0. 
            use this x to solve dot n = 0, i.e. n_out = e^(-1/x). if n_in and n_out are the same,
            its solved. otherwise, make a new guess for n_in and repeat until n_out = n_in to 
            within tolerances.
        """

        print('\n  iter  n_out    x_0      convergence')
        print('--------------------------------------')

        converged = False
        for iter in range(max_iter):

            # solve for x(n). 
            x_0 = self._solve_dot_U_steady_state_constant_j(n_in)

            # calculate n(x)
            n_out = self._calc_n_constant_j(x_0,n_in)

            residual = np.abs(n_out-n_in)

            print(f'{iter:5d}   {n_out:6.4f}   {x_0:6.4f}   {residual:5.2e}')

            # check convergence
            if residual <= n_tol:
                converged = True
                break

            # make new guess
            n_in = self._simple_mixing(n_in,n_out,alpha)

        if not converged:
            print('\n*** WARNING ***\nfailed to converge!\n')

        return n_out, x_0, residual

    # ----------------------------------------------------------------------------------------------

    def _solve_dot_U_steady_state_constant_j(self,n):

        """
        solve 0 = j^2 x / n + y^4 - x^4 for x
        """

        _x = self.x
        _y = self.y

        _j_sq = self.j_sq

        _f = _j_sq * _x / n + _y**4 - _x**4 
        _inds, _ = self._find_zeros(_f)

        _zeros = _x[_inds]
        if _zeros.size > 1:
            print('fuck')
            print(_zeros)
            
        x_0 = _zeros[0]

        return x_0
    
    # ----------------------------------------------------------------------------------------------

    def _solve_dot_U_steady_state_constant_v(self,n):

        """
        solve 0 = v^2 n / x + y^4 - x^4 for x
        """

        _x = self.x
        _y = self.y

        _v_sq = self.v_sq

        _f = _v_sq * n / _x + _y**4 - _x**4 
        _inds, _ = self._find_zeros(_f)

        x_0 = _x[_inds]

        return x_0
    
    # ----------------------------------------------------------------------------------------------

    def _calc_n_constant_j(self,x_0,n):

        """
        solve n = e^(-1/x) e^(j*z/n)
        """

        _j = self.j
        _z = self.z

        return np.exp(-1/x_0) * np.exp(_j*_z/n)
    
    # ----------------------------------------------------------------------------------------------

    def _calc_n_constant_v(self,x_0):

        """
        solve n = e^(-1/x) e^(j*z/n)
        """

        _v = self.v
        _z = self.z

        return np.exp(-1/x_0) * np.exp( _v*_z / x_0 )

    # ----------------------------------------------------------------------------------------------

    def _find_zeros(self,arr):

        """
        find the 0's in the array arr.
        """
        
        diff = np.diff(np.sign(arr))
        zeros = np.flatnonzero(diff)
        
        return zeros, diff
    
    # ----------------------------------------------------------------------------------------------

    def _simple_mixing(self,n_in,n_out,alpha):

        """
        simple mixing: n^(i+1)_in = alpha * n^i_out + (1-alpha) * n^i_in
        """ 

        return alpha * n_out + (1-alpha) * n_in

    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

def run_v_sweep(y=0.1,z=0.1):

    """
    sweep over v
    """

    num_v = 1001
    v = np.linspace(0.0,0.5,num_v)

    n = np.zeros((num_v,3),dtype=float)
    x = np.zeros((num_v,3),dtype=float)

    count = 0
    for ii, vv in enumerate(v):

        print(f'\ncount: {count}')
        
        bistable = c_bistable_defects(y=y,z=z,x_hi=1.0,num_x=10001)
        _n, _x  = bistable.solve_constant_v(vv)
        
        if _n.size == 1:
            n[ii,0] = _n.squeeze()
            x[ii,0] = _x.squeeze()
        else:
            n[ii,:] = _n
            x[ii,:] = _x
            
        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_v_sweep_y_{y:.3f}_z_{z:.3f}.h5','w') as db:

        db.create_dataset('n',data=n)
        db.create_dataset('x',data=x)
        
        db.create_dataset('v',data=v)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

def run_j_sweep(y=0.1,z=0.1):

    """
    sweep over v
    """

    num_j = 101
    j = np.linspace(0.0,0.01,num_j)

    n_lo = np.zeros(num_j,dtype=float)
    x_lo = np.zeros(num_j,dtype=float)
    n_hi = np.zeros(num_j,dtype=float)
    x_hi = np.zeros(num_j,dtype=float)

    count = 0
    for ii, jj in enumerate(j):

        print(f'\ncount: {count}')
        
        bistable = c_bistable_defects(y=y,z=z,x_hi=1.0,num_x=10001)
        n_lo[ii], x_lo[ii], n_hi[ii], x_hi[ii] = bistable.solve_constant_j(jj)

        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_j_sweep_y_{y:.3f}_z_{z:.3f}.h5','w') as db:

        db.create_dataset('n_lo',data=n_lo)
        db.create_dataset('x_lo',data=x_lo)
        db.create_dataset('n_hi',data=n_hi)
        db.create_dataset('x_hi',data=x_hi)
        
        db.create_dataset('j',data=j)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    z=0.01

    # run_v_sweep(y=0.01,z=z)
    # run_v_sweep(y=0.1,z=z)
    # run_v_sweep(y=0.25,z=z)

    run_j_sweep(y=0.1,z=z)
    run_j_sweep(y=0.175,z=z)
    run_j_sweep(y=0.25,z=z)

