
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

    def __init__(self,beta=6,gamma=1):

        self.comm, self.rank, self.num_ranks = get_mpi_world()

        self.beta = beta
        self.gamma = gamma

        self.y_critical = ( beta/gamma + 1 ) ** (- ( beta + gamma )/( beta*gamma ) )
        self.z_critical = np.sqrt( beta/gamma * ( gamma * np.e / 
                                    ( gamma + beta ) ) ** ( ( gamma + beta ) / gamma ) )
        
    # ----------------------------------------------------------------------------------------------

    def _evaluate_func(self,x,y,z):

        _b = self.beta
        _g = self.gamma

        return np.exp(1 / x ** _g) * (x**_b - y**_b) - z**2

    # ----------------------------------------------------------------------------------------------

    def _get_zeros(self,f):

        return np.flatnonzero(np.diff(np.sign(f)))
        
    # ----------------------------------------------------------------------------------------------

    def get_bistability_region(self,nx=1000,ny=1000,nz=1000,
                               xlo=None,xhi=None,ylo=None,yhi=None,zlo=None,zhi=None):
        
        if xhi is None:
                xhi = ( (self.gamma + self.beta) / self.gamma ) ** ( 1/ self.gamma )

        if ylo is None:
            ylo = 1e-2
        if yhi is None:
            yhi = self.y_critical * 1.1

        if zlo is None:
            zlo = 1e-2
        if zhi is None:
            zhi = self.z_critical * 5.0

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
            _f = self._evaluate_func(_x,_y,_z)
            _zeros = self._get_zeros(_f)

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

        ax.annotate(r'T$_{critical}$',xy=(self.y_critical,self.z.max()*1.025),
                    xycoords='data',c='r',annotation_clip=False)
        ax.annotate(r'V$_{critical}$',xy=(self.y.max()*1.025,self.z_critical*0.99),
                    xycoords='data',c='r',annotation_clip=False)
        
        ax.annotate(r'$\gamma$'+f'={self.gamma}, '+r'$\beta$'+f'={self.beta}',
                    xy=(0.05,0.95),xycoords='axes fraction',c='r',annotation_clip=False)

        ax.set_ylabel('z (V)')
        ax.set_xlabel(r'y (T$_{ph}$)')
        plt.savefig(f'bistability_region_gamma_{self.gamma}_beta_{self.beta}.png',dpi=300, 
                    bbox_inches='tight')
        
        plt.show()

    # ----------------------------------------------------------------------------------------------


        

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    bistable = c_bistable(beta=4)
    bistable.get_bistability_region(ny=2500,nz=2500)
    bistable.plot_bistability_region()


# ny = 1000
# nz = 1000
# y = np.linspace(0.01,0.25,ny)
# z = np.linspace(0.01,1,nz)

# yarr, zarr = np.meshgrid(y,z,indexing='ij')
# yarr = yarr.flatten()
# zarr = zarr.flatten()

# if rank == 0:
#     bistable = np.zeros((nz,ny),dtype=int).flatten()
# else:
#     bistable = None


# split = np.array_split(np.arange(bistable.size),size)

# my_bistable = np.zeros((nz//size,ny),dtype=int)
# my_y = yarr[split[rank]]
# my_z = zarr[split[rank]]

# exit()

# for ii, zz in enumerate(z):
#     print(ii)

#     for jj, yy in enumerate(y):

#         x = np.linspace(yy,1,100)

#         F = x ** beta - yy ** beta - zz ** 2 * np.exp( - 1 / x ** gamma )
#         zeros = np.flatnonzero(np.diff(np.sign(F)))
        
#         if zeros.size > 1:
#             # plt.plot(x,F)
#             # plt.ylim(F.min()*1.1,-F.min()*1.1)
#             # plt.show()
#             bistable[ii,jj] = 1

# fig, ax = plt.subplots(figsize=(4.5,4.5))

# ax.imshow(bistable, origin='lower', extent=(y.min(), y.max(), z.min(), z.max()), 
#           aspect='auto', cmap='binary', vmin=0, vmax=1, interpolation='none')

# z0 = np.sqrt( beta/gamma * 
#              ( gamma * np.e / ( gamma + beta ) ) ** ( ( gamma + beta ) / gamma ) )
# ax.axhline(z0,lw=1,color='r')


# ax.set_xlabel('y')
# ax.set_ylabel('z')
# ax.set_title('Bistability Region')
# plt.show()

