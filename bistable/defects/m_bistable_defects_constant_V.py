
import numpy as np
import h5py

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,v=0.2,y=0.1,j=1000,z=0.0,x_lo=None,x_hi=5,num_x=10001):

        """
        dot U = 0 = v^2 n + x ( y^4 - x^4 )
        dot n = 0 = e^(-1/x) - n

        v is dimensionless voltage, x is dimensionless sample temperature, y is 
            dimensionless bath temperature. n is defect concentration.

        v and y are "knobs" that we turn in the lab. x and n are variables we solve for. 
        """

        self.v = v
        self.vsq = v**2

        self.j = j
        self.jsq = j

        self.z = z
        self.y = y

        if z > 1/v:
            exit('it is required that z < 1/v')

        if x_lo is None:
            x_lo = y
        self.x = np.linspace(x_lo,x_hi,num_x)

    # ----------------------------------------------------------------------------------------------

    def solve(self):

        """
        ...
        """

        # low n guess
        x0, n0 = self._solve()
        j0 = self._calc_current(x0,n0)

        if x0.size > 1:
            mulitstable = True
        else: 
            mulitstable = False

        v = self.v
        y = self.y
        z = self.z

        print('y:',y)
        print('v:',v)
        print('z:',z)
        print('x0:',x0)
        print('n0:',n0)

        return x0, n0, j0, v ,y ,z

    # ----------------------------------------------------------------------------------------------

    def _solve(self):

        """
        solve 0 = v^2 n / x + y^4 - x^4 for x
            with n = e^(-(1-vz)/x) 
        """

        _x = self.x
        _y = self.y
        _vsq = self.vsq

        _n = self._calc_n(_x)

        _f = _vsq * _n / _x + ( _y**4 - _x**4 )
        _inds, _ = self._find_zeros(_f)

        x0 = _x[_inds]
        n0 = self._calc_n(x0)

        return x0, n0
    
    # ----------------------------------------------------------------------------------------------

    def _calc_n(self,x_0):

        """
        solve n = e^(-(1-vz)/x)
        """

        _v = self.v
        _z = self.z

        return np.exp(-(1 - _v * _z )/x_0)
    
    # ----------------------------------------------------------------------------------------------

    def _calc_current(self,n,x):

        """
        solve v = j x/n => j = v n / x
        """

        return self.v * n / x

    # ----------------------------------------------------------------------------------------------

    def _find_zeros(self,arr):

        """
        find the 0's in the array arr.
        """
        
        diff = np.diff(np.sign(arr))
        zeros = np.flatnonzero(diff)
        
        return zeros, diff
    
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
            
            bistable = c_bistable_defects(v=vv,y=yy)
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

def run_v_sweep(y=0.025,z=0.5):

    """
    sweep over v
    """

    nv = 1001
    v = np.linspace(0.0,0.5,nv)

    n = np.zeros((nv,3),dtype=float)
    x = np.zeros((nv,3),dtype=float)
    j = np.zeros((nv,3),dtype=float)

    count = 0
    for ii, vv in enumerate(v):

        print(f'\ncount: {count}')
        
        bistable = c_bistable_defects(v=vv,y=y,z=z,x_hi=1.0,num_x=10001)
        x0, n0, j0, _, _, _ = bistable.solve()

        print(x0)

        if x0.size > 1:
            x[ii,...] = x0
            n[ii,...] = n0
            j[ii,...] = j0
        else:
            x[ii,0] = x0.squeeze()
            n[ii,0] = n0.squeeze()
            j[ii,0] = j0.squeeze()

        count += 1

    # write results to hdf5 file
    with h5py.File(f'results_v_sweep_y_{y:.3f}.h5','w') as db:

        db.create_dataset('n',data=n)
        db.create_dataset('x',data=x)
        db.create_dataset('v',data=v)
        db.create_dataset('y',data=y)
        db.create_dataset('z',data=z)

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # run_sweep()

    run_v_sweep(y=0.01)
    run_v_sweep(y=0.1)
    run_v_sweep(y=0.25)
    run_v_sweep(y=0.5)

    # bistable = c_bistable_defects(v=0.22,y=0.1,x_hi=1.0)
    # bistable.solve()
        
