
import numpy as np
import h5py

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,v=0.2,y=0.1,x_lo=None,x_hi=5,num_x=10001):

        """
        dot U = 0 = v^2 n + x ( y^4 - x^4 )
        dot n = 0 = e^(-1/x) - n

        v is dimensionless voltage, x is dimensionless sample temperature, y is 
            dimensionless bath temperature. n is defect concentration.

        v and y are "knobs" that we turn in the lab. x and n are variables we solve for. 
        """

        self.v = v
        self.vsq = v**2
        self.y = y

        if x_lo is None:
            x_lo = y
        self.x = np.linspace(x_lo,x_hi,num_x)

    # ----------------------------------------------------------------------------------------------

    def solve(self,n_lo_guess=0.0,n_hi_guess=1.0,max_iter=1000,n_tol=1e-9,alpha=0.4,count=None):

        """
        we look for two solutions for n (there may be more that we miss!). we have to solve 
            self-consistently, so how we find different solutions is we have to make a guess
            close the different solutions. we just pick n->0 and n->1 and solve for each. if they
            converge to the same solution, there is only one. if they are different, there is
            multistability!
        """

        # low n guess
        print(f'\nsolving for low n guess: n_in={n_lo_guess:6.4f}')
        n_lo, x_lo = self._solve_self_consistent(n_lo_guess,max_iter,n_tol,alpha)

        # hi n guess
        print(f'\nsolving for high n guess: n_in={n_hi_guess:6.4f}')
        n_hi, x_hi = self._solve_self_consistent(n_hi_guess,max_iter,n_tol,alpha)

        n_diff = n_hi-n_lo
        x_diff = x_hi-x_lo

        msg = '\n*** RESULTS ***'
        if count is not None:
            msg += f'\ncount: {count}'
        msg += f'\n% n_lo: {n_lo:9.6e}'
        msg += f'\n% x_lo: {x_lo:9.6e}'
        msg += f'\n% n_hi: {n_hi:9.6e}'
        msg += f'\n% x_hi: {x_hi:9.6e}'
        msg += f'\n% n_diff: {n_diff:9.6f}'
        msg += f'\n% x_diff: {x_diff:9.6f}'
        msg += f'\n% v: {self.v:9.6f}'
        msg += f'\n% y: {self.y:9.6f}'
        print(msg)

        return n_lo, x_lo, n_hi, x_hi, n_diff, x_diff

    # ----------------------------------------------------------------------------------------------

    def _solve_self_consistent(self,n_in,max_iter,n_tol,alpha):
        
        """
        solve for n self consistently: make a guess for n_in and calculate x from dot U = 0. 
            use this x to solve dot n = 0, i.e. n_out = e^(-1/x). if n_in and n_out are the same,
            its solved. otherwise, make a new guess for n_in and repeat until n_out = n_in to 
            within tolerances.
        """

        print('\n  iter   n_in    n_out   convergence')
        print('--------------------------------------')

        converged = False
        for iter in range(max_iter):

            # solve for x(n). 
            x_0 = self._solve_dot_U_0(n_in)

            # solve n=exp(-1/x(n))
            n_out = self._solve_dot_n_0(x_0)
            residual = np.abs(n_out-n_in)

            print(f'{iter:5d}   {n_in:6.4f}   {n_out:6.4f}   {residual:5.2e}')

            # check convergence
            if residual <= n_tol:
                converged = True
                break

            # make new guess
            n_in = np.copy(self._simple_mixing(n_in,n_out,alpha))

        if not converged:
            print('\n*** WARNING ***\nfailed to converge!\n')
        else:
            print(f'\nconverged!\nn={n_out:6.4f}, x={x_0:6.4f}')

        return n_out, x_0

    # ----------------------------------------------------------------------------------------------

    def _solve_dot_U_0(self,n):

        """
        solve 0 = v^2 n + x ( y^4 - x^4 )for x
        """

        _x = self.x
        _y = self.y
        _vsq = self.vsq

        _f = _vsq * n + _x * ( _y**4 - _x**4 )
        _inds, _ = self._find_zeros(_f)

        _zeros = _x[_inds]
        if _zeros.size > 1:
            print('fuck')
            print(_zeros)
            
        x0 = _zeros[0]

        return x0
    
    # ----------------------------------------------------------------------------------------------

    def _solve_dot_n_0(self,x_0):

        """
        solve n = e^(-1/x)
        """

        return np.exp(-1/x_0)

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

def run_sweep():

    """
    sweep over y and y
    """

    ny = 101
    nv = 101
    y = np.linspace(0.0,0.25,ny)
    v = np.linspace(0.0,0.5,nv)

    multistable = np.zeros((ny,nv),dtype=bool)
    n_diff = np.zeros((ny,nv),dtype=float)
    x_diff = np.zeros((ny,nv),dtype=float)
    n_lo = np.zeros((ny,nv),dtype=float)
    x_lo = np.zeros((ny,nv),dtype=float)
    n_hi = np.zeros((ny,nv),dtype=float)
    x_hi = np.zeros((ny,nv),dtype=float)

    count = 0
    for ii, yy in enumerate(y):
        for jj, vv in enumerate(v):

            print(f'\ncount: {count}')
            
            bistable = c_bistable_defects(vv,yy)
            multistable[ii,jj], n_lo[ii,jj], x_lo[ii,jj], n_hi[ii,jj], x_hi[ii,jj], \
                n_diff[ii,jj], x_diff[ii,jj] = bistable.solve(count=count)

            count += 1

    # write results to hdf5 file
    with h5py.File('results.h5','w') as db:

        db.create_dataset('multistable',data=multistable)
        db.create_dataset('n_lo',data=n_lo)
        db.create_dataset('x_lo',data=x_lo)
        db.create_dataset('n_hi',data=n_hi)
        db.create_dataset('x_hi',data=x_hi)
        db.create_dataset('n_diff',data=n_diff)
        db.create_dataset('x_diff',data=x_diff)
        db.create_dataset('y',data=y)
        db.create_dataset('v',data=v)

# --------------------------------------------------------------------------------------------------

def run_v_sweep(y=0.025):

    """
    sweep over v
    """

    nv = 1001
    v = np.linspace(0.0,1.0,nv)

    n_diff = np.zeros(nv,dtype=float)
    x_diff = np.zeros(nv,dtype=float)
    n_lo = np.zeros(nv,dtype=float)
    x_lo = np.zeros(nv,dtype=float)
    n_hi = np.zeros(nv,dtype=float)
    x_hi = np.zeros(nv,dtype=float)

    count = 0
    for ii, vv in enumerate(v):

        print(f'\ncount: {count}')
        
        bistable = c_bistable_defects(vv,y,x_hi=2.0,num_x=10001)
        n_lo[ii], x_lo[ii], n_hi[ii], x_hi[ii], n_diff[ii], x_diff[ii] = \
            bistable.solve(count=count)

        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_v_sweep_y_{y:.3f}.h5','w') as db:

        db.create_dataset('n_lo',data=n_lo)
        db.create_dataset('x_lo',data=x_lo)
        db.create_dataset('n_hi',data=n_hi)
        db.create_dataset('x_hi',data=x_hi)
        db.create_dataset('n_diff',data=n_diff)
        db.create_dataset('x_diff',data=x_diff)
        db.create_dataset('y',data=y)
        db.create_dataset('v',data=v)

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # run_sweep()
    run_v_sweep(y=0.01)
    run_v_sweep(y=0.1)
    run_v_sweep(y=0.25)
    run_v_sweep(y=0.5)
    run_v_sweep(y=1.0)
        
