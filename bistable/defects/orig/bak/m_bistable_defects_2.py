
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

def find_zeros_inds(arr):

    """
    find the indices where zeros (roots) occur. root is bracketed by ind, ind+1
    """
    
    diff = np.diff(np.sign(arr))
    inds = np.flatnonzero(diff)

    return inds

# --------------------------------------------------------------------------------------------------

def find_zeros_adaptive(func,x,args=(),x_tol=1e-16,max_iter=100,verbose=False):

    """
    find multiple zeros of a function to high precision. uses scipy.optimize.root_scalar
    """

    # print(x)

    arr = func(x,*args)
    inds = find_zeros_inds(arr)
    num_zeros = inds.size

    if verbose:
        msg = f'\nfound {num_zeros} zeros'
        print(msg)

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

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,y=0.1,z=0.0,x_lo=None,x_hi=1e9,num_x=1001):

        """
        y is dimensionless bath temp, z is dimensionless field lowering parameter ... 
        will explain better later when less lazy
        """

        self.y = y
        self.z = z

        if x_lo is None:
            x_lo = y
        self.x = np.linspace(x_lo,x_hi,num_x)

        self.num_x = num_x

        print('\n##############################################################')

    # ----------------------------------------------------------------------------------------------

    def _find_n_root(self,n):

        """
        ...
        """

        x0 = np.zeros(n.size)
        for ii, nn in enumerate(n):

            _x0 = find_zeros_adaptive(self._calc_dot_U_constant_j,x=self.x,args=(nn,))
            if _x0.size == 0:
                msg = '\n*** ERROR ***\nno solution for x_0. try increasing x_hi.\n'
                msg += f'n: {nn}'
                print(msg)
                exit()

            x0[ii] = _x0

        arr = np.exp( -1/x0 ) * np.exp( self.j*self.z / n ) - n
        inds = find_zeros_inds(arr)
        n0 = n[inds]

        return n0

    # ----------------------------------------------------------------------------------------------

    def solve_constant_j_n_grid(self,j,num_n=1001):

        """
        ...
        """

        self.j = j 
        self.j_sq = j**2

        n_arr = np.logspace(-6,0,num_n)
        n0 = find_zeros_adaptive(func=self._find_n_root,x_tol=1e-6,x=n_arr,verbose=True)
        
        print(n0)

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

    def solve_constant_j(self,j,n_guess=0.1,max_iter=200,n_tol=1e-9):

        """
        ...
        """

        self.max_iter = max_iter
        self.n_tol = n_tol

        self.j = j 
        self.j_sq = j**2

        print(f'\nconstant j: {j:9.6e}')
        print(f'\nsolving for n_guess = {n_guess:9.6e}')

        n_0, x_0 = self._solve_constant_j(n_guess)

        msg = '\n*** RESULTS ***'
        msg += f'\n\n% j: {self.j:9.6e}'
        msg += f'\n% y: {self.y:9.6e}'
        msg += f'\n% z: {self.z:9.6e}' 
        msg += f'\n% n_0: {n_0:9.6e}'
        msg += f'\n% x_0: {x_0:9.6e}'
        print(msg)

        return n_0, x_0
    
    # ----------------------------------------------------------------------------------------------

    def _solve_constant_j(self,n_guess):

        """
        ...
        """

        n_out, _result = scipy.optimize.newton(func=self._const_j_kernel,x0=n_guess,
                                                    tol=self.n_tol,disp=False,full_output=True,
                                                    maxiter=self.max_iter)
        x_out = find_zeros_adaptive(self._calc_dot_U_constant_j,x=self.x,args=(n_out,))

        if not _result.converged:
            print('\n*** WARNING ***\nscipy.optimize newton failed to converge')
            n_out = np.nan
            x_out = np.nan

        print('')
        print(_result)

        return n_out, x_out

    # ----------------------------------------------------------------------------------------------

    def _const_j_kernel(self,n):

        """
        a kernel for newton method
        """

        # _eps = 1e-16
        # if n < _eps:
        #     n = 1e-16
        # if n > 1-_eps:
        #     n = 1-_eps

        _x_0 = find_zeros_adaptive(self._calc_dot_U_constant_j,x=self.x,args=(n,))
        
        if _x_0.size == 0:
            msg = '\n*** ERROR ***\nno solution for x_0. try increasing x_hi.\n'
            msg += f'n: {n}'
            print(msg)
            exit()
    
        return self._calc_n_constant_j(_x_0,n)-n

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

def run_j_n_grid(y=0.1,z=0.1):

    """
    run a calculation for an array of j
    """

    bistable = c_bistable_defects(y=y,z=z,x_hi=1e9,num_x=1001)

    num_j = 1001
    j = np.linspace(0.0,0.1,num_j)

    count = 0
    for ii, _j in enumerate(j):

        print(f'\nnow doing count: {count}')
        bistable.solve_n_grid(j=_j,num_n=1001)
        count += 1

    # # write results to hdf5 file
    # with h5py.File(f'results_j_y_{y:.3f}_z_{z:.3f}_n_sweep.h5','w') as db:

    #     db.create_dataset('n',data=n_solutions)
    #     db.create_dataset('x',data=x_solutions)
    #     db.create_dataset('j',data=j)
    #     db.create_dataset('y',data=y)
    #     db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

def run_j_guess_array(y=0.1,z=0.1):

    """
    run a calculation for an array of j
    """

    _n = int(1e4+1)
    bistable = c_bistable_defects(y=y,z=z,x_hi=1e6,num_x=_n)

    num_j = 1001
    j = np.linspace(0.0,0.01,num_j)

    n_guess = [0.1,0.5]
    num_guesses = len(n_guess)

    n_solutions = np.zeros((num_j,num_guesses))
    x_solutions = np.zeros((num_j,num_guesses))

    count = 0
    for ii, _j in enumerate(j):
        for jj, _n in enumerate(n_guess):

            print(f'\nnow doing count: {count}')
            n_solutions[ii,jj], x_solutions[ii,jj] = bistable.solve_constant_j(_j,n_guess=_n,
                                                                    n_tol=1e-9)
            count += 1

    # write results to hdf5 file
    with h5py.File(f'results_j_y_{y:.3f}_z_{z:.3f}_n_sweep.h5','w') as db:

        db.create_dataset('n',data=n_solutions)
        db.create_dataset('x',data=x_solutions)
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
        # run_j(y,z)
        run_j_guess_array(y,z)

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # run_j_n_grid(y=0.1,z=0.0)

    bistable = c_bistable_defects(y=0.1,z=0.0)
    bistable.solve_constant_j_n_grid(j=0.05)

    # z=0.10
    # run_j(0.1,z)
    # run_v(y=0.1,z=z)

    # y_list = [0.001,0.005,0.010,0.050,0.100,0.250,0.500]
    # # z_list = [0.0,0.001,0.005,0.010,0.050,0.100]

    # y_list = [0.1]
    # z_list = [0.1]

    # for zz in z_list:
    #     run_j_sweep_over_y(y_list,zz)
