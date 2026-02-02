
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import h5py


# --------------------------------------------------------------------------------------------------

class _c_zeros:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):

        self.zeros = []
        self.num_zeros = 0
        self.queue = []
        self.num_queue = 0
        self.queue_iter = 0

    # ----------------------------------------------------------------------------------------------

    def get_zeros(self):

        return np.array(self.zeros)

    # ----------------------------------------------------------------------------------------------

    def add_zero(self,zero):

        self.zeros.append(zero)
        self.num_zeros += 1

    # ----------------------------------------------------------------------------------------------

    def add_to_queue(self,x,inds):

        for ind in inds:
            self._add_to_queue([x[ind],x[ind+1]])

    # ----------------------------------------------------------------------------------------------

    def _add_to_queue(self,bounds):

        self.queue.append(bounds)
        self.num_queue += 1

    # ----------------------------------------------------------------------------------------------

    def get_from_queue(self):

        self.num_queue += -1
        return self.queue.pop(0)

    # ----------------------------------------------------------------------------------------------

    def print(self):

        print('\nqueue_iter:',self.queue_iter)
        print('queue:\n',np.array(self.queue))
        print('zeros:',np.array(self.zeros),'\n')

    # ----------------------------------------------------------------------------------------------

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

def find_zeros_adaptive(func,x,args=(),x_tol=1e-16,max_iter=100):

    """
    find all of the zeros in an array to high precision. find the indices of zeros in rough array
    first, then loop over the found zeros and refine each using scipy.optimize.root_scalar to high
    precision
    """

    # find the coarse bounds for zeros
    arr = func(x,*args)
    inds = find_zeros_inds(arr)
    num_zeros = inds.size

    if num_zeros == 0:
        msg = '\n*** ERROR ***\nno zeros found'
        print(msg)
        exit()

    zeros = np.zeros(num_zeros)
    
    # loop over the zeros and refine them
    for ii, ind in enumerate(inds):
        
        x_lo = x[ind]
        x_hi = x[ind+1]
        
        res = scipy.optimize.root_scalar(lambda x: func(x,*args), method='bisect', 
                                         bracket=[x_lo,x_hi], xtol=x_tol, maxiter=max_iter)

        zeros[ii] = res.root

    return np.atleast_1d(zeros)

# --------------------------------------------------------------------------------------------------

def find_zeros_inds(arr):

    """
    find the indices of all the zeros in the array. nb: the zeros are bounded by ind, ind+1
    """
        
    # diff = np.diff(np.sign(arr))
    # return np.flatnonzero(diff)

    return np.flatnonzero( np.sign(arr[:-1]) * np.sign(arr[1:]) < 0 )

# --------------------------------------------------------------------------------------------------

def bisect(func,x_lo,x_hi,args,zeros,x_tol,max_iter,num_x=1001,r_tol=None):

    """
    ...
    """

    x_ref = (x_lo+x_hi) / 2.0

    if r_tol is None:
        r_tol = x_tol

    for ii in range(max_iter):

        x = np.linspace(x_lo,x_hi,num_x)
        arr = func(x,*args)

        ind = find_zeros_inds(arr)

        if ind.size == 0:
            msg = '\n*** ERROR ***\nbisection failed, no zeros found'
            print(msg)
            zeros.print()
            exit()

        # if found multiple zeros, add the new bounds to queue and start over
        if ind.size > 1:
            zeros.add_to_queue(x,ind)
            return None
                
        ind = ind.squeeze()
        
        x_lo = x[ind]
        x_hi = x[ind+1]
        x_0 = (x_lo + x_hi) / 2.0

        if np.isclose(x_0,x_ref,atol=x_tol,rtol=r_tol):
            return x_0
        
        x_ref = x_0
        
    msg = '\n*** WARNING ***\nfailed to converge!'
    print(msg)

    return x_0

# --------------------------------------------------------------------------------------------------

def find_zeros_adaptive_custom(func,x,args=(),x_tol=1e-12,max_iter=100,verbose=False):

    """
    ...
    """

    # find the coarse bounds for zeros
    arr = func(x,*args)
    inds = find_zeros_inds(arr)

    # debugging
    if verbose:
        _num_zeros = inds.size
        msg = f'\nfound {_num_zeros} zeros'
        for ii in range(_num_zeros):
            msg += f'\n  {ii}: x0 = {x[inds[ii]]}'
        print(msg)
    
    # object to store the zeros we're working on
    zeros = _c_zeros()
    zeros.add_to_queue(x,inds)

    # loop over all the zeros in the queue 
    while True:

        # zeros.print()

        num_queue = zeros.num_queue
        if num_queue == 0:
            break
        
        # remove zero interval from queue
        x_lo, x_hi = zeros.get_from_queue()
        
        # refine zero 
        result = bisect(func=func,x_lo=x_lo,x_hi=x_hi,args=args,zeros=zeros,
                        x_tol=x_tol,max_iter=max_iter)

        # if found multiple zeros, go back and start over        
        if result is None:
            continue

        zeros.add_zero(result)
    
    return zeros.get_zeros()

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

class c_bistable:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,y,z,x_lo=None,x_hi=1e9,num_x=1001):

        """
        const v:
            0 = v^2 n / x + y^4 - x^4 
            n = e^(-1/x) e^(vz/x)

        const j:
            0 = j^2 x / n + y^4 - x^4 
            n = e^(-1/x) e^(jz/n)
        """

        print('\n##############################################################')

        self.y = y
        self.z = z
        
        if x_lo is None:
            x_lo = self.y
        self.x = np.linspace(x_lo,x_hi,num_x)

    # ----------------------------------------------------------------------------------------------

    def solve_constant_v(self,v):

        """
        need to solve
            n = e^(-(1-vz)/x(n))
        with x(n) given by the solution of
            0 = v^2 n / x + y^4 - x^4
        """

        self.v = v 
        self.v_sq = v**2

        msg = f'\ndoing constant v'
        msg += f'\nj: {self.v}'
        msg += f'\ny: {self.y}'
        msg += f'\nz: {self.z}' 
        print(msg)
        
        # # old version
        # x_0 = _find_zeros_adaptive(self._calc_dot_U_const_v, x=self.x)
        # n_0 = self._calc_n_const_v(x_0)

        # solve x(n)
        if self.v == 0.0:
            x_0 = np.array([self.y])
        else:
            x_0 = find_zeros_adaptive(self._calc_dot_U_const_v, x=self.x) 
            
        # plug x(n) into dot n = 0
        n_0 = self._calc_n_const_v(x_0)

        # print results
        _num_zeros = n_0.size
        msg = f'\nnum_zeros: {_num_zeros}'
        for ii in range(_num_zeros):
            msg += f'\n\n solution {ii}:'
            msg += f'\n    n: {n_0[ii]}'
            msg += f'\n    x: {x_0[ii]}'
        print(msg,flush=True)  
            
        return n_0, x_0

    # ----------------------------------------------------------------------------------------------

    def solve_constant_j(self,j,num_n=1001):

        """
        need to solve
            n = e^(-1/x(n)) e^(jz/n)
        with x(n) given by the solution of
            0 = j^2 x / n + y^4 - x^4
        """
        
        self.j = j
        self.j_sq = j**2

        msg = f'\ndoing constant j'
        msg += f'\nj: {self.j}'
        msg += f'\ny: {self.y}'
        msg += f'\nz: {self.z}' 
        print(msg)

        if self.j == 0.0:
            x_0 = np.array([self.y])
            self.v = 0.0
            n_0 = self._calc_n_const_v(x_0)
        else:        
            n_lo = max(self.j*self.z*self.y,1e-24)
            n = np.linspace(n_lo,1,num_n)
            # n_0 = find_zeros_adaptive_custom(func=self._calc_n_balance_const_j,x=n)
            n_0 = find_zeros_adaptive(func=self._calc_n_balance_const_j,x=n)
            x_0 = find_zeros_adaptive(self._calc_dot_U_const_j,x=self.x,args=(n_0,))

        # print results
        _num_zeros = n_0.size
        msg = f'\nnum_zeros: {_num_zeros}'
        for ii in range(_num_zeros):
            msg += f'\n\n solution {ii}:'
            msg += f'\n    n: {n_0[ii]}'
            msg += f'\n    x: {x_0[ii]}'
        print(msg,flush=True)   

        return n_0, x_0
    
    # ----------------------------------------------------------------------------------------------

    def solve_constant_j_newton(self,j,n_guess,n_tol=1e-9,max_iter=5000):

        """
        need to solve
            n = e^(-1/x(n)) e^(jz/n)
        with x(n) given by the solution of
            0 = j^2 x / n + y^4 - x^4
        """

        self.j = j
        self.j_sq = j**2

        msg = f'\ndoing constant j'
        msg += f'\nj: {self.j}'
        msg += f'\ny: {self.y}'
        msg += f'\nz: {self.z}' 
        msg += '\n\nusing newton-raphson method'
        msg += f'\nn_guess: {n_guess}'
        print(msg)

        if self.j == 0.0:

            x_0 = self.y
            self.v = 0.0
            n_0 = self._calc_n_const_v(x_0)

        else:

            n_0, _result = scipy.optimize.newton(func=self._calc_n_balance_const_j,x0=n_guess,
                                                tol=n_tol,disp=False,full_output=True,
                                                maxiter=max_iter,rtol=n_tol)
            x_0 = find_zeros_adaptive(self._calc_dot_U_const_j,x=self.x,args=(n_0,)).squeeze()

            if not _result.converged:
                print('\n*** WARNING ***\nscipy.optimize newton failed to converge')
                n_0 = np.atleast_1d(np.nan)
                x_0 = np.atleast_1d(np.nan)

            print('')
            print(_result)

        # print results
        msg = f'\nn: {n_0}'
        msg += f'\nx: {x_0}'
        print(msg,flush=True)   

        return n_0, x_0
        
    # ----------------------------------------------------------------------------------------------

    def _calc_n_balance_const_j(self,n):

        """
        fill the x(n) array and calculate 
            f(n) = e^(-1/x(n)) e^(jz/n) - n
        """

        _n = np.atleast_1d(n)

        _x0 = np.zeros(_n.size)
        for ii, nn in enumerate(_n):

            # fill x(n)
            _x = find_zeros_adaptive(self._calc_dot_U_const_j,x=self.x,args=(nn,)).squeeze()

            if _x.size > 1:
                msg = '\n*** ERROR ***\nfound multiple zeros'
                print(msg)
                exit()

            _x0[ii] = _x

        _x0 = _x0.squeeze()
        return np.exp(-1/_x0) * np.exp(self.j * self.z / n) - n
    
    # ----------------------------------------------------------------------------------------------

    def _calc_dot_U_const_j(self,x,n):

        """
        0 = j^2 x / n + y^4 - x^4
        """

        # if n < np.exp(-1/self.y):
        #     n = np.exp(-1/self.y)

        return self.j_sq * x / n + (self.y**4 - x**4)

    # ----------------------------------------------------------------------------------------------

    def _calc_dot_U_const_v(self,x):

        """
        0 = j^2 x / n + y^4 - x^4
        """

        _n = self._calc_n_const_v(x)
        return self.v_sq * _n / x + (self.y**4 - x**4)
    
    # ----------------------------------------------------------------------------------------------

    def _calc_n_const_v(self,x):

        """
        n = e^(-1/x) e^( v * z / x)
        """

        return np.exp( -(1 - self.v*self.z) / x )

    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

def run_v(y=0.1,z=0.1):

    """
    run calculation for an array of v
    """

    num_v = 10001
    v = np.linspace(0.0,0.5,num_v)

    n = np.zeros((num_v,3),dtype=float)
    x = np.zeros((num_v,3),dtype=float)

    count = 0
    for ii, vv in enumerate(v):

        print(f'\nnow doing count: {count}')
        
        bistable = c_bistable(y=y,z=z,x_hi=10,num_x=1001)
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

def run_v_sweep_over_y(y_list=[0.01,0.1,0.25],z=0.0):

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
    run calculation for an array of v
    """

    num_j = 10001
    j = np.linspace(0.0,0.1,num_j)

    n = np.zeros((num_j,3),dtype=float)
    x = np.zeros((num_j,3),dtype=float)

    count = 0
    for ii, jj in enumerate(j):

        print(f'\nnow doing count: {count}')
        
        bistable = c_bistable(y=y,z=z,x_hi=1e9,num_x=1001)
        _n, _x  = bistable.solve_constant_j(jj)
        
        if _n.size == 1:
            n[ii,0] = _n.squeeze()
            x[ii,0] = _x.squeeze()
        else:
            n[ii,:] = _n
            x[ii,:] = _x
            
        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_j_y_{y:.3f}_z_{z:.3f}.h5','w') as db:

        db.create_dataset('n',data=n)
        db.create_dataset('x',data=x)
        
        db.create_dataset('j',data=j)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

def run_j_newton(y=0.1,z=0.1):

    """
    run calculation for an array of v
    """

    num_j = 251
    j = np.linspace(0.0,0.1,num_j)

    n_guess = [1e-9,1e-6,1e-3,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.999]
    num_n = len(n_guess)

    n = np.zeros((num_j,num_n),dtype=float)
    x = np.zeros((num_j,num_n),dtype=float)

    count = 0
    for ii, jj in enumerate(j):
        for kk, nn in enumerate(n_guess):
                
            print(f'\nnow doing count: {count}')
            
            bistable = c_bistable(y=y,z=z,x_hi=1e9,num_x=1001)
            _n, _x  = bistable.solve_constant_j_newton(jj,nn)
            
            n[ii,kk] = _n
            x[ii,kk] = _x
            
            count += 1

    # write results to hdf5 file
    with h5py.File(f'results_j_y_{y:.3f}_z_{z:.3f}_newton.h5','w') as db:

        db.create_dataset('n',data=n)
        db.create_dataset('x',data=x)
        
        db.create_dataset('j',data=j)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

def run_j_sweep_over_y(y_list=[0.01,0.1,0.25],z=0.0):

    """
    sweep over multiple y for the same j 
    """

    comm, proc, num_procs = get_mpi()

    my_y = np.array_split(y_list,num_procs)[proc]
    for y in my_y:

        print('\nproc:',proc,'\ty:',y)

        run_j(y,z)
        # run_j_newton(y,z)

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # run_j(y=0.1,z=0.1)
    # run_j_newton(y=0.1,z=0.1)
    # run_j_sweep_over_y()

    # bistable = c_bistable(y=0.01,z=0.0,x_hi=1e3,num_x=1001)
    # bistable.solve_constant_j_newton(j=0.0004,n_guess=0.1)

    # y_list = [0.01,0.1,0.25]
    # z_list = [0.0,0.1,1.0,5.0]

    y_list = [0.01,0.1,0.25]
    z_list = [0.0,0.1,1.0]

    for zz in z_list:
        run_j_sweep_over_y(y_list,zz)
        run_v_sweep_over_y(y_list,zz)
