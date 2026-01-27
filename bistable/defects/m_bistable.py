
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
        self.iter = 0

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

        self.iter += 1
        np.savetxt(f'bounds_iter_{self.iter}',np.array(self.queue),fmt='%.24f')

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

        print('\nqueue:\n',np.array(self.queue))
        print('zeros:',np.array(self.zeros))
    
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

def _find_zeros_adaptive(func,x,args=(),x_tol=1e-16,max_iter=100,verbose=False):

    """
    find all of the zeros in an array to high precision. find the indices of zeros in rough array
    first, then loop over the found zeros and refine each using scipy.optimize.root_scalar to high
    precision
    """

    # find the coarse bounds for zeros
    arr = func(x,*args)
    inds = find_zeros_inds(arr)
    num_zeros = inds.size

    if verbose:

        msg = f'\nfound {num_zeros} zeros'
        for ii in range(num_zeros):
            msg += f'\n  {ii}: x0 = {x[inds[ii]]}'
        print(msg)

    zeros = np.zeros(num_zeros)
    
    # loop over the zeros and refine them
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

    return zeros

# --------------------------------------------------------------------------------------------------

def find_zeros_inds(arr):

    """
    find the indices of all the zeros in the array. nb: the zeros are bounded by ind, ind+1
    """

    _sign = np.sign(arr)
    inds = np.flatnonzero( np.sign(arr[:-1]) * np.sign(arr[1:]) < 0 )

    return inds

# --------------------------------------------------------------------------------------------------

def bisect(func,x_lo,x_hi,args,zeros,x_tol,max_iter,num_x=1001,r_tol=None):

    """
    ...
    """

    if r_tol is None:
        r_tol = x_tol

    for ii in range(max_iter):

        x = np.linspace(x_lo,x_hi,num_x)
        arr = func(x,*args)
        # plt.plot(x,arr)
        # plt.show()

        ind = find_zeros_inds(arr)

        # if found multiple zeros, add the new bounds to queue and start over
        if ind.size > 1:
            zeros.add_to_queue(x,ind)
            return None
                
        ind = ind.squeeze()
        
        x_0 = x[ind]
        x_lo = x[ind]
        x_hi = x[ind+1]

        if np.abs(x_hi-x_lo) <= x_tol + r_tol * max(x_lo,x_hi):
            return x_0
        
    msg = '\n*** WARNING ***\nfailed to converge!'
    print(msg)

    return x_0

# --------------------------------------------------------------------------------------------------

def find_zeros_adaptive(func,x,args=(),x_tol=1e-12,max_iter=100,verbose=False):

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

    def __init__(self,y,z,x_lo=None,x_hi=1,num_x=1001):

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

        print(f'doing constant v: v = {v}')

        # x_0 = _find_zeros_adaptive(self._calc_dot_U_const_v, x=self.x)
        # n_0 = self._calc_n_const_v(x_0)

        if self.v == 0.0:
            x_0 = np.array([self.y])
        else:
            x_0 = find_zeros_adaptive(self._calc_dot_U_const_v, x=self.x)
            
        n_0 = self._calc_n_const_v(x_0)

        _num_zeros = n_0.size
        msg = '\n*** RESULTS ***'
        msg += f'\n\nv: {self.v}'
        msg += f'\ny: {self.y}'
        msg += f'\nz: {self.z}' 
        msg += f'\n\nnum_zeros: {_num_zeros}\n'
        for ii in range(_num_zeros):
            msg += f'\n solution {ii} :'
            msg += f'\n    n = {n_0[ii]}'
            msg += f'\n    x = {x_0[ii]}\n'
        print(msg)
            
        return n_0, x_0

    # ----------------------------------------------------------------------------------------------

    def solve_constant_j(self,j,num_n=1001):

        """
        need to solve
            n = e^(-1/x(n)) e^(jz/n)
        with x(n) given by the solution of
            0 = j^2 x / n + y^4 - x^4
        """

        print(f'doing constant j: j = {j}')

        self.j = j
        self.j_sq = j**2

        if self.j == 0.0:
            x_0 = np.array([self.y])
            n_0 = self._calc_n_const_v(x_0)
        else:
            n = np.logspace(-24,0,num_n)
            n0 = find_zeros_adaptive(func=self._calc_n_balance_const_j,x=n)
            print(n0)
        
    # ----------------------------------------------------------------------------------------------

    def _calc_n_balance_const_j(self,n):

        """
        fill the x(n) array and find the zeros of 
            f(n) = e^(-1/x(n)) e^(jz/n) - n
        """

        x = np.zeros(n.size)
        for ii, nn in enumerate(n):

            _x = find_zeros_adaptive(self._calc_dot_U_const_j,x=self.x,args=(nn,)).squeeze()

            if _x.size > 1:
                msg = '\n*** ERROR ***\nfound multiple zeros'
                print(msg)
                exit()
            else:
                _x = np.nan

            x[ii] = _x

        return np.exp(-1/x) * np.exp(self.j * self.z / n) - n
    
    # ----------------------------------------------------------------------------------------------

    def _calc_dot_U_const_j(self,x,n):

        """
        0 = j^2 x / n + y^4 - x^4
        """

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

        return np.exp( - (1 - self.v*self.z) / x )

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
        
        bistable = c_bistable(y=y,z=z,x_hi=1.0,num_x=10001)
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

if __name__ == '__main__':

    bistable = c_bistable(y=0.1,z=0.1)
    bistable.solve_constant_j(0.05)

    # bistable = c_bistable(y=0.01,z=0.0)
    # bistable.solve_constant_v(0.218)

    # run_v_sweep_over_y()
