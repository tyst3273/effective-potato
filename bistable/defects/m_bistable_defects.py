
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI


# --------------------------------------------------------------------------------------------------

def get_mpi_world():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    return comm, rank, num_ranks

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

class c_bistable_defects:

    # ----------------------------------------------------------------------------------------------

    def __init__(self):

        self.comm, self.rank, self.num_ranks = get_mpi_world()

    # ----------------------------------------------------------------------------------------------

    def solve(self):

        """
        dot U = 0 = v^2 / ( x + y ) + z^4 - y^4
        dot n = 0 = gamma e^(-1/y) - x => x = gamma e^(-1/y(x))

        v is dimensionless voltage, y is dimensionless sample temperature, z is dimensionless
            bath temperature, and x is dimensionless defect concentration. 

        v and z are "knobs" that we turn in the lab. x and y are variables we solve for. 
            gamma is a parameter. 

        x = gamma n ; 0 <= n <= 1. then 0 <= x <= gamma .

        make an initial guess for x. solve dot U for y(x). plug y(x) into x = gamma e^(-1/y(x)).
            call x' = gamma e^(-1/y(x)). if x' != x, make a new guess for x by "mixing" x and x'.
            then calculate y(x) again, calculate x', and repeat this until x and x' are the same
            within some predefined tolerance.

        nb: need to keep track of multiple solutions for y(x) because if multiple branches appear 
            for y (ie multistability), we need to keep all branches.
        """

        v = 1.0
        z = 0.1
        x_init = 0.0001

        y = np.linspace(0,6,1000)

        f = v**2 / (x_init+y) + z**4 - y**4
        plt.plot(y,f,c='b')
        plt.axhline(0,c='k')
        
        plt.show()

    # ----------------------------------------------------------------------------------------------

        

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    bistable = c_bistable_defects()
    bistable.solve()













