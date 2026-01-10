
import numpy as np

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,v=0.2,y=0.1,x_lo=None,x_hi=10,num_x=10000):

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

    def solve(self,n_lo=1e-3,n_hi=0.999,max_iter=1000,n_tol=1e-9,alpha=0.4,degeneracy_tol=1e-3):

        """
        we look for two solutions for n (there may be more that we miss!). we have to solve 
            self-consistently, so how we find different solutions is we have to make a guess
            close the different solutions. we just pick n->0 and n->1 and solve for each. if they
            converge to the same solution, there is only one. if they are different, there is
            multistability!
        """

        # low n guess
        print(f'\nsolving for low n guess: n_in={n_lo:6.4f}')
        n_lo, x_lo = self._solve_self_consistent(n_lo,max_iter,n_tol,alpha)

        # hi n guess
        print(f'\nsolving for high n guess: n_in={n_hi:6.4f}')
        n_hi, x_hi = self._solve_self_consistent(n_hi,max_iter,n_tol,alpha)

        if np.abs(n_lo-n_hi) > degeneracy_tol:
            multistable = True
        else:
            multistable = False

        msg = '\n*** RESULTS ***'
        msg += f'\n% multistable: {multistable}'
        msg += f'\n% n_lo: {n_lo:9.6e}'
        msg += f'\n% x_lo: {x_lo:9.6e}'
        msg += f'\n% n_hi: {n_hi:9.6e}'
        msg += f'\n% x_hi: {x_hi:9.6e}'
        msg += f'\n% n_diff={n_hi-n_lo:9.6f}'
        msg += f'\n% x_diff={x_hi-x_lo:9.6f}'
        msg += f'\n% v: {self.v:9.6f}'
        msg += f'\n% y: {self.y:9.6f}'
        print(msg)

    # ----------------------------------------------------------------------------------------------

    def _solve_self_consistent(self,n_in=1e-3,max_iter=1000,n_tol=1e-6,alpha=0.4):
        
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
            n_in = self._simple_mixing(n_in,n_out,alpha)

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
            print(_zeros.size)
            exit('fuck')
        x0 = _zeros.squeeze()

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

if __name__ == '__main__':

    # bistable = c_bistable_defects()
    # bistable.solve()

    y = np.linspace(0.0,5,501)
    v = np.linspace(0.0,5,501)

    count = 0
    for yy in y:
        for vv in v:

            print(f'\ncount: {count}')
            
            bistable = c_bistable_defects(vv,yy)
            bistable.solve()

            count += 1












