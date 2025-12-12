
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

class c_bistable:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,beta=6,gamma=1,alpha=10):

        self.comm, self.rank, self.num_ranks = get_mpi_world()

        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha

        self.y_critical = ( beta/gamma + 1 ) ** (- ( beta + gamma )/( beta*gamma ) )
        self.z_critical = np.sqrt( beta/gamma * ( gamma * np.e / 
                                    ( gamma + beta ) ) ** ( ( gamma + beta ) / gamma ) )
        
    # ----------------------------------------------------------------------------------------------

    def _evaluate_Fx(self,x,y,z):

        _b = self.beta
        _g = self.gamma
        _a = self.alpha

        return  z**2 * np.exp(-1/x) * (1+_a*_g*np.exp((1-_g)/x)) + y**_b - x**_b

    # ----------------------------------------------------------------------------------------------

    def _get_zeros(self,f):
        
        diff = np.diff(np.sign(f))
        zeros = np.flatnonzero(diff)
        
        return zeros, diff
        
    # ----------------------------------------------------------------------------------------------

    def get_bistability_region(self,nx=1000,ny=1000,nz=1000,
                               xlo=None,xhi=None,ylo=None,yhi=None,zlo=None,zhi=None):
        
        if xhi is None:
                xhi = ( (self.gamma + self.beta) / self.gamma ) ** ( 1/ self.gamma ) 

        if ylo is None:
            ylo = 1e-3
        if yhi is None:
            yhi = 0.2 # self.y_critical * 1.05

        if zlo is None:
            zlo = 0.0 #self.z_critical * 0.95
        if zhi is None:
            zhi = 0.5 #self.z_critical * 2.0

        y = np.linspace(ylo,yhi,ny)
        z = np.linspace(zlo,zhi,nz)

        _y_grid, _z_grid = np.meshgrid(y,z,indexing='ij')
        _shape = _y_grid.shape
        _size = _y_grid.size
        
        _y_grid = _y_grid.flatten()
        _z_grid = _z_grid.flatten()

        _rank = self.rank
        _num_ranks = self.num_ranks
        _split = np.array_split(np.arange(_size),_num_ranks)
        _my_inds = _split[_rank]
        _my_count = _my_inds.size

        if _rank == 0:
            bistable = np.zeros(_size,dtype=int)
        else:
            bistable = None

        if _num_ranks == 1:
            _my_bistable = bistable
        else:
            _my_bistable = np.zeros(_my_count,dtype=int)

        for ii, _ind in enumerate(_my_inds):

            if _rank == 0:
                if ii % 1000 == 0:
                    print(f'now on {ii} out of {_my_count}',flush=True)

            _y = _y_grid[_ind]
            _z = _z_grid[_ind]

            if xlo is None:
                xlo = _y

            _x = np.linspace(xlo,xhi,nx)
            _f = self._evaluate_Fx(_x,_y,_z)
            _zeros, _diff = self._get_zeros(_f)

            if _zeros.size > 1:
                _my_bistable[ii] = 1

        if _num_ranks > 1:

            if _rank == 0:

                bistable[_my_inds] = _my_bistable
                for _rr in range(1,_num_ranks):
                    _inds = _split[_rr]
                    _num = _inds.size
                    _bistable = np.zeros(_num,dtype=int)
                    self.comm.Recv(_bistable,source=_rr,tag=1)
                    bistable[_inds] = _bistable

            else:
                self.comm.Send(_my_bistable,dest=0,tag=1)

        if _rank == 0:

            bistable.shape = _shape

            self.bistable = bistable
            self.y = y
            self.z = z      

    # ----------------------------------------------------------------------------------------------

    def plot_bistability_region(self):

        if self.rank != 0:
            return
        
        fig, ax = plt.subplots(figsize=(4.5,4.5))

        extent = [self.y.min(),self.y.max(),self.z.min(),self.z.max()]
        ax.imshow(self.bistable.T,aspect='auto',origin='lower',extent=extent,cmap='binary',
                  interpolation='none')
        
        ax.axvline(self.y_critical,lw=1,c='r')
        ax.axhline(self.z_critical,lw=1,c='r')

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1) #,direction='in')
        ax.tick_params(which='major',length=5)
        ax.set_rasterization_zorder = 1000000000

        ax.annotate(r'y$_{critical}$',xy=(self.y_critical,self.z.max()*1.025),
                    xycoords='data',c='r',annotation_clip=False)
        ax.annotate(r'z$_{critical}$',xy=(self.y.max()*1.025,self.z_critical*0.99),
                    xycoords='data',c='r',annotation_clip=False)
        
        ax.annotate(r'$\gamma$'+f'={self.gamma}, '+r'$\beta$'+f'={self.beta}',
                    xy=(0.05,0.95),xycoords='axes fraction',c='r',annotation_clip=False)

        ax.set_ylabel('z(V)')
        ax.set_xlabel(r'y(T$_{ph}$)')
        # plt.savefig(f'bistability_region_gamma_{self.gamma}_beta_{self.beta}.png',dpi=300, 
        #             bbox_inches='tight')
        
        plt.show()

    # ----------------------------------------------------------------------------------------------
        

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    bistable = c_bistable(gamma=1,beta=6)
    
    bistable.get_bistability_region(nx=5000,ny=1000,nz=1000)
    bistable.plot_bistability_region()
    
