
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy

# --------------------------------------------------------------------------------------------------

def get_mpi():

    """
    self explanatory
    """

    import mpi4py.MPI as mpi
    comm = mpi.COMM_WORLD
    proc = comm.Get_rank()
    num_procs = comm.Get_size()

    return comm, proc, num_procs

# --------------------------------------------------------------------------------------------------

def find_zeros_adaptive(func,x,args=(),x_tol=1e-16,max_iter=100,verbose=False):

    """
    find multiple zeros of a function to high precision. uses scipy.optimize.root_scalar
    """

    arr = func(x,*args)
    diff = np.diff(np.sign(arr))
    inds = np.flatnonzero(diff)
    num_zeros = inds.size

    zeros = np.zeros(num_zeros)
        
    for ii, ind in enumerate(inds):
        
        x_lo = x[ind]
        x_hi = x[ind+1]
        
        res = scipy.optimize.root_scalar(lambda x: func(x,*args), method='bisect', 
                                         bracket=[x_lo,x_hi], xtol=x_tol, maxiter=max_iter)
        
        if verbose:
            print('')
            print(res)
            print('')

        zeros[ii] = res.root

    return zeros.squeeze()

# --------------------------------------------------------------------------------------------------

class _c_anderson_mixer:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,num_history):

        """
        find multiple zeros of a function to high precision. uses scipy.optimize.root_scalar
        """

        pass
    
    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,y=0.1,z=0.0,x_lo=None,x_hi=1,num_x=100001):

        """
        y is dimensionless bath temp, z is dimensionless field lowering parameter ... 
        will explain better later when less lazy
        """

        self.y = y
        self.z = z

        if x_lo is None:
            x_lo = y
        self.x = np.linspace(x_lo,x_hi,num_x)

        # self.x_lo = x_lo
        # self.x_hi = x_hi
        self.num_x = num_x

        print('\n##############################################################')

    # ----------------------------------------------------------------------------------------------

    def solve_constant_v(self,v):

        """
        ...
        """

        self.v = v 
        self.v_sq = v**2
        print(f'\nconstant v: {v:9.6e}')

        x_0 = find_zeros_adaptive(self._calc_dot_U_constant_v, x=self.x)
        n_0 = self._calc_n_constant_v(x_0)

        msg = '\n*** RESULTS ***'
        msg += f'\n\n% v: {self.v:9.6e}'
        msg += f'\n% y: {self.y:9.6e}'
        msg += f'\n% z: {self.z:9.6e}' 
        print(msg)

        print('\n% n:',n_0)
        print('% x:',x_0)

        return n_0, x_0
    
    # ----------------------------------------------------------------------------------------------

    def _solve_constant_j(self,n_in):

        """
        ...
        """

        n_out, _result = scipy.optimize.newton(func=self._const_j_kernel,x0=n_in,
                                                    tol=self.n_tol,disp=False,full_output=True,
                                                    maxiter=self.max_iter)
        x_out = find_zeros_adaptive(self._calc_dot_U_constant_j,x=self.x,args=(n_in,))

        _conv = _result.converged
        if not _conv:
            print('\n*** WARNING ***\nscipy.optimize newton failed to converge')
        print('')
        print(_result)

        return n_out, x_out
    
    # ----------------------------------------------------------------------------------------------

    def solve_constant_j_for_guess_array(self,j,n_lo=0.001,n_hi=0.999,num_n=11,max_iter=200,
                                         n_tol=1e-9,alpha=0.4,method='anderson'):
        
        """
        ...
        """

        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_tol = n_tol

        self.j = j 
        self.j_sq = j**2
        print(f'\nconstant j: {j:9.6e}')

        _n_array = np.linspace(n_lo,n_hi,num_n)
        n_solutions = np.zeros(num_n)
        x_solutions = np.zeros(num_n)

        for ii, _nn in enumerate(_n_array):

            print(f'\nsolving for n guess: n = {_nn:9.6e}')
            _n_out, _x_out = self._solve_constant_j(_nn)
            print(_x_out)

            n_solutions[ii] = _n_out
            x_solutions[ii] = _x_out

        _prec = np.abs(np.floor(np.log10(n_tol))).astype(int)
        _n_unique, _inds = np.unique(np.round(n_solutions,_prec+1),return_index=True)
        n_solutions = n_solutions[_inds]
        x_solutions = x_solutions[_inds]

    # ----------------------------------------------------------------------------------------------

    def solve_constant_j(self,j,n_lo_guess=0.001,n_hi_guess=0.999,max_iter=200,n_tol=1e-9,
                         alpha=0.4,method='anderson'):

        """
        we look for two solutions for n. we have to solve self-consistently, so how we find 
            different solutions is we have to make a guess close the different solutions. 
            we just pick n->0 and n->1 and solve for each. if they converge to the same solution, 
            there is only one. if they are different, there is multistability!
        """

        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_tol = n_tol

        self.j = j 
        self.j_sq = j**2
        print(f'\nconstant j: {j:9.6e}')

        # newtons method
        if self.method == 'newton':
            
            print(f'\nsolving for low n guess: n_in = {n_lo_guess:9.6e}')
            n_lo, x_lo = self._solve_constant_j(n_lo_guess)

            print(f'\nsolving for high n guess: n_in = {n_hi_guess:9.6e}')
            n_hi, x_hi = self._solve_constant_j(n_hi_guess)

        else:

            # low n guess
            print(f'\nsolving for low n guess: n_in = {n_lo_guess:9.6e}')
            n_lo, x_lo, res_lo = \
                self._solve_self_consistent_constant_j(n_lo_guess)

            # hi n guess
            print(f'\nsolving for high n guess: n_in = {n_hi_guess:9.6e}')
            n_hi, x_hi, res_hi = \
                self._solve_self_consistent_constant_j(n_hi_guess)

        n_diff = n_hi-n_lo
        x_diff = x_hi-x_lo

        msg = '\n*** RESULTS ***'
        msg += f'\n\n% j: {self.j:9.6e}'
        msg += f'\n% y: {self.y:9.6e}'
        msg += f'\n% z: {self.z:9.6e}' 

        msg += f'\n\n% n_lo: {n_lo:9.6e}'
        msg += f'\n% x_lo: {x_lo:9.6e}'
        # msg += f'\n% residual_lo: {res_lo:9.6e}'

        msg += f'\n\n% n_hi: {n_hi:9.6e}'
        msg += f'\n% x_hi: {x_hi:9.6e}'
        # msg += f'\n% residual_hi: {res_hi:9.6e}'

        msg += f'\n\n% n_diff: {n_diff:9.6e}'
        msg += f'\n% x_diff: {x_diff:9.6e}'
        print(msg)

        return n_lo, x_lo, n_hi, x_hi
    
    # ----------------------------------------------------------------------------------------------

    def _const_j_kernel(self,n_in):

        """
        a kernel for newton method
        """

        if n_in < 1e-16:
            n_in = 1e-16

        _x_0 = find_zeros_adaptive(self._calc_dot_U_constant_j,x=self.x,args=(n_in,))

        if _x_0.size == 0:
            msg = '\n*** ERROR ***\nno solution for x_0. increase x_max.'
            print(msg)
            exit()
    
        return self._calc_n_constant_j(_x_0,n_in)-n_in

    # ----------------------------------------------------------------------------------------------

    def _solve_self_consistent_constant_j(self,n_in):
        
        """
        solve for n self consistently: make a guess for n_in and calculate x from dot U = 0. 
            use this x to solve dot n = 0, i.e. n_out = e^(-1/x). if n_in and n_out are the same,
            its solved. otherwise, make a new guess for n_in and repeat until n_out = n_in to 
            within tolerances.
        """

        _max_iter = self.max_iter
        _n_tol = self.n_tol

        if self.method == 'anderson':
            _mixer = _c_anderson_mixer()

        print('\n  iter   n_out          x_0           convergence')
        print('------------------------------------------------------')

        converged = False
        for iter in range(_max_iter):

            # solve for x(n)
            x_0 = find_zeros_adaptive(self._calc_dot_U_constant_j,x=self.x,args=(n_in,))
            if x_0.size > 1:
                exit('fuck')

            # calculate n(x)
            n_out = self._calc_n_constant_j(x_0,n_in)

            residual = n_out-n_in

            print(f'{iter:5d}   {n_out:9.6e}   {x_0:9.6e}   {residual:9.6e}')

            # check convergence
            if np.abs(residual) <= _n_tol:
                converged = True
                break

            # make new guess
            if self.method == 'anderson':
                n_in = _mixer.mix()
            else:
                n_in = self._simple_mixing(n_in,n_out)

        if not converged:
            print('\n*** WARNING ***\nfailed to converge!\n')

        return n_out, x_0, residual
    
    # ----------------------------------------------------------------------------------------------

    def _calc_dot_U_constant_j(self,x,n):

        """
        energy balance is j^2 * x / n + y^4 - x^4 
        """

        _j_sq = self.j_sq
        _y = self.y

        return _j_sq * x / n + _y**4 - x**4
    
    # ----------------------------------------------------------------------------------------------

    def _calc_dot_U_constant_v(self,x):

        """
        energy balance is v^2 * n / x + y^4 - x^4 
        """

        _v_sq = self.v_sq
        _y = self.y

        _n = self._calc_n_constant_v(x)
        return _v_sq * _n / x + _y**4 - x**4
    
    # ----------------------------------------------------------------------------------------------

    def _calc_n_constant_j(self,x,n):

        """
        solve n = e^(-1/x) e^(j*z/n)
        """

        _j = self.j
        _z = self.z

        return np.exp(-1/x) * np.exp(_j*_z/n)
    
    # ----------------------------------------------------------------------------------------------

    def _calc_n_constant_v(self,x):

        """
        solve n = e^(-1/x) e^(j*z/n)
        """

        _v = self.v
        _z = self.z

        return np.exp(-1/x) * np.exp(_v*_z/x)
    
    # ----------------------------------------------------------------------------------------------

    def _simple_mixing(self,n_in,n_out):

        """
        simple mixing: n^(i+1)_in = alpha * n^i_out + (1-alpha) * n^i_in
        """ 

        _a = self.alpha

        return _a * n_out + (1-_a) * n_in

    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

def run_v(y=0.1,z=0.1):

    """
    run calculation for an array of v
    """

    num_v = 1001
    v = np.linspace(0.0,0.5,num_v)

    n = np.zeros((num_v,3),dtype=float)
    x = np.zeros((num_v,3),dtype=float)

    count = 0
    for ii, vv in enumerate(v):

        print(f'\nnow doing count: {count}')
        
        bistable = c_bistable_defects(y=y,z=z,x_hi=1.0,num_x=1001)
        _n, _x  = bistable.solve_constant_v(vv)
        
        if _n.size == 1:
            n[ii,0] = _n.squeeze()
            x[ii,0] = _x.squeeze()
        else:
            n[ii,:] = _n
            x[ii,:] = _x
            
        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_v_y_{y:.3f}_z_{z:.3f}.h5','w') as db:

        db.create_dataset('n',data=n)
        db.create_dataset('x',data=x)
        
        db.create_dataset('v',data=v)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

def run_v_sweep_over_y(y_list=[0.0,0.1],z=0.0):

    """
    sweep over multiple y for the same j 
    """

    comm, proc, num_procs = get_mpi()

    my_y = np.array_split(y_list,num_procs)[proc]
    for y in my_y:
        print('\nproc:',proc,'\ty:',y)
        run_v(y,z)

# --------------------------------------------------------------------------------------------------

def run_j(y=0.1,z=0.1):

    """
    run a calculation for an array of j
    """

    num_j = 1001
    j = np.linspace(0.0,0.1,num_j)
    # j = np.logspace(-6,0,num=num_j)

    n_lo = np.zeros(num_j,dtype=float)
    x_lo = np.zeros(num_j,dtype=float)
    n_hi = np.zeros(num_j,dtype=float)
    x_hi = np.zeros(num_j,dtype=float)

    count = 0
    for ii, jj in enumerate(j):

        print(f'\nnow doing count: {count}')
        
        bistable = c_bistable_defects(y=y,z=z,x_hi=10.0,num_x=10001)

        # n_lo[ii], x_lo[ii], n_hi[ii], x_hi[ii] = bistable.solve_constant_j(jj,
        #                             n_lo_guess=0.001,n_hi_guess=0.2,n_tol=1e-9,method='newton')
        
        n_lo[ii], x_lo[ii], n_hi[ii], x_hi[ii] = \
                        bistable.solve_constant_j_for_guess_array(jj,n_tol=1e-9,method='newton')

        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_j_y_{y:.3f}_z_{z:.3f}.h5','w') as db:

        db.create_dataset('n_lo',data=n_lo)
        db.create_dataset('x_lo',data=x_lo)
        db.create_dataset('n_hi',data=n_hi)
        db.create_dataset('x_hi',data=x_hi)
        
        db.create_dataset('j',data=j)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

def run_j_sweep_over_y(y_list=[0.0,0.1],z=0.0):

    """
    sweep over multiple y for the same j 
    """

    comm, proc, num_procs = get_mpi()

    my_y = np.array_split(y_list,num_procs)[proc]
    for y in my_y:
        print('\nproc:',proc,'\ty:',y)
        run_j(y,z)

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # z=0.10
    # run_j(0.1,z)
    # run_v(y=0.1,z=z)

    y_list = [0.001,0.005,0.010,0.050,0.100,0.250,0.500]
    z_list = [0.0,0.001,0.005,0.010,0.050,0.100]

    y_list = [0.05,0.075,0.1,0.125,0.15]
    # z_list = [0.1]

    # for zz in z_list:
    #     run_v_sweep_over_y(y_list,zz)

    for zz in z_list:
        run_j_sweep_over_y(y_list,zz)
