
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

    def _evaluate_Lx(self,x,y,z):

        _b = self.beta
        _g = self.gamma

        return  -(z**2 * np.exp(-1 / x ** _g) + y**_b - x**_b)

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
            ylo = 1e-2
        if yhi is None:
            yhi = self.y_critical * 1.05

        if zlo is None:
            zlo = self.z_critical * 0.95
        if zhi is None:
            zhi = self.z_critical * 2.0

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
            _f = self._evaluate_Lx(_x,_y,_z)
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
        plt.savefig(f'bistability_region_gamma_{self.gamma}_beta_{self.beta}.png',dpi=300, 
                    bbox_inches='tight')
        
        # plt.show()

    # ----------------------------------------------------------------------------------------------

    def solve_bistability_fixed_z(self,frac,nx=1000,ny=1000,xlo=None,xhi=None,ylo=None,yhi=None):
        
        _z = frac * self.z_critical
        
        if xhi is None:
                xhi = ( (self.gamma + self.beta) / self.gamma ) ** ( 1/ self.gamma ) * 2

        if ylo is None:
            ylo = 0.0
        if yhi is None:
            yhi = self.y_critical * 1.05

        y = np.linspace(ylo,yhi,ny)
        _size = y.size

        solutions = [[] for _ in range(_size)]
        slopes = [[] for _ in range(_size)]
        
        for ii in range(_size):

            if ii % 1000 == 0:
                print(f'now on {ii} out of {_size}',flush=True)

            _y = y[ii]

            if xlo is None:
                xlo = 0

            _x = np.linspace(xlo,xhi,nx)
            _f = self._evaluate_Lx(x=_x,y=_y,z=_z)
            _zeros, _diff = self._get_zeros(_f)
            
            solutions[ii].extend(_x[_zeros])
            slopes[ii].extend(_diff[_zeros])

        self.solutions = solutions
        self.slopes = slopes
        self.y = y
        self.z = _z

    # ----------------------------------------------------------------------------------------------

    def plot_solutions_fixed_z(self):

        if self.rank != 0:
            return
        
        fig, ax = plt.subplots(figsize=(4.5,4.5))

        _min = 1e6
        _max = 0.0

        _size = self.y.size
        for ii in range(_size):

            _sol = self.solutions[ii]
            _y = self.y[ii]

            for jj, _x in enumerate(_sol):
                if self.slopes[ii][jj] > 0:
                    ax.plot(_y,_x,marker='o',ms=1.5,c='k')
                else:
                    ax.plot(_y,_x,marker='o',ms=1.5,c='m')

            _ = min(_sol)
            if _ < _min:
                _min = _
            _ = max(_sol)
            if _ > _max:
                _max = _

        ax.plot(self.y,self.y,lw=1.0,ls='--',c='b')

        # _z = self.z
        # _b = self.beta
        # _g = self.gamma
        # _y = self.y

        # # linearized solution
        # _L = - _z**2 * np.exp(-_y**(-_g))
        # _Lp = _b * _y **(_b-1) - _z**2 * _g * _y **(-_g-1) * np.exp(-_y**(-_g))
        # _Lpp = _b * (_b - 1) * _y **(_b-2) - (_z * _g * _y **(-_g-1)) **2 * np.exp(-_y**(-_g)) \
        #     + _z**2 * _g * (_g+1) * _y**(-_g-2) * np.exp(-_y**(-_g))
        
        # # f1 = _y - _L/_Lp
        # # ax.plot(self.y,f1,c='g')

        # _c2 = _Lpp / 2 
        # _c1 = _Lp - _Lpp * _y
        # _c0 = _Lpp / 2 * _y**2 - _Lp * _y + _L

        # f2p =  ( -_c1 + np.sqrt(_c1**2 - 4*_c2*_c0) ) / 2 / _c2
        # f2m =  ( -_c1 - np.sqrt(_c1**2 - 4*_c2*_c0) ) / 2 / _c2
        # ax.plot(self.y,f2p,c='g')
        # # ax.plot(self.y,f2m,c='g')

        # _x_inf = self.z ** (2/self.beta) 
        # ax.axhline(_x_inf,lw=1,c='r')

        ax.axvline(self.y_critical,lw=1,c='r')
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1) #,direction='in')
        ax.tick_params(which='major',length=5)
        ax.set_rasterization_zorder = 1000000000

        ax.axis([self.y.min(),self.y.max(),0,_max*1.05])

        ax.annotate(r'y$_{critical}$',xy=(self.y_critical*0.95,_max*1.075),
                    xycoords='data',c='r',annotation_clip=False)
        ax.annotate(r'x=y',xy=(self.y.max()*1.025,self.y_critical*1.01),
                    xycoords='data',c='b',annotation_clip=False)
        
        ax.annotate(r'$\gamma$'+f'={self.gamma}, '+r'$\beta$'+f'={self.beta}',
                    xy=(0.05,0.85),xycoords='axes fraction',c='r',annotation_clip=False)
        
        _frac = self.z / self.z_critical
        ax.annotate(f'z={_frac:.3f}'+r'$\times$z$_0$',xy=(0.05,0.8),xycoords='axes fraction',
                    c='r',annotation_clip=False)

        ax.set_xlabel(r'y(T$_{ph}$)')
        ax.set_ylabel(r'x(T$_{e}$)')

        plt.savefig(f'bistability_soltions_{self.gamma}_beta_{self.beta}_z0_{_frac:.3f}.png',
                    dpi=300, bbox_inches='tight')
        
        # plt.show()

    # ----------------------------------------------------------------------------------------------

    def solve_bistability_fixed_y(self,frac,nx=1000,nz=1000,xlo=None,xhi=None,zlo=None,zhi=None):
        
        _y = frac * self.y_critical
        
        if xhi is None:
                xhi = ( (self.gamma + self.beta) / self.gamma ) ** ( 1/ self.gamma ) * 2

        if zlo is None:
            zlo = self.z_critical * 0
        if zhi is None:
            zhi = self.z_critical * 4.0
            
        z = np.linspace(zlo,zhi,nz)
        _size = z.size

        solutions = [[] for _ in range(_size)]
        slopes = [[] for _ in range(_size)]
        
        for ii in range(_size):

            if ii % 1000 == 0:
                print(f'now on {ii} out of {_size}',flush=True)

            _z = z[ii]

            if xlo is None:
                xlo = 0

            _x = np.linspace(xlo,xhi,nx)
            _f = self._evaluate_Lx(x=_x,y=_y,z=_z)
            _zeros, _diff = self._get_zeros(_f)
            
            solutions[ii].extend(_x[_zeros])
            slopes[ii].extend(_diff[_zeros])

        self.solutions = solutions
        self.slopes = slopes
        self.y = _y
        self.z = z

    # ----------------------------------------------------------------------------------------------
    
    def plot_solutions_fixed_y(self):

        if self.rank != 0:
            return
        
        fig, ax = plt.subplots(figsize=(4.5,4.5))

        _min = 1e6
        _max = 0.0

        _size = self.z.size
        for ii in range(_size):

            _sol = self.solutions[ii]
            _z = self.z[ii]

            for jj, _x in enumerate(_sol):
                if self.slopes[ii][jj] > 0:
                    ax.plot(_z,_x,marker='o',ms=1.5,c='k')
                else:
                    ax.plot(_z,_x,marker='o',ms=1.5,c='m')

            _ = min(_sol)
            if _ < _min:
                _min = _
            _ = max(_sol)
            if _ > _max:
                _max = _

        ax.axvline(self.z_critical,lw=1,c='r')
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1) #,direction='in')
        ax.tick_params(which='major',length=5)
        ax.set_rasterization_zorder = 1000000000

        ax.axis([self.z.min(),self.z.max(),0,_max*1.05])

        ax.annotate(r'z$_{critical}$',xy=(self.z_critical*0.95,_max*1.075),
                    xycoords='data',c='r',annotation_clip=False)
        
        ax.annotate(r'$\gamma$'+f'={self.gamma}, '+r'$\beta$'+f'={self.beta}',
                    xy=(0.3,0.85),xycoords='axes fraction',c='r',annotation_clip=False)
        
        _frac = self.y / self.y_critical
        ax.annotate(f'y={_frac:.3f}'+r'$\times$y$_0$',xy=(0.3,0.8),xycoords='axes fraction',
                    c='r',annotation_clip=False)

        ax.set_xlabel(r'z(V)')
        ax.set_ylabel(r'x(T$_{e}$)')

        plt.savefig(f'bistability_soltions_{self.gamma}_beta_{self.beta}_y0_{_frac:.3f}.png',
                    dpi=300, bbox_inches='tight')
        
        # plt.show()

    # ----------------------------------------------------------------------------------------------
    
    def plot_IV_fixed_y(self):

        if self.rank != 0:
            return
        
        fig, ax = plt.subplots(figsize=(4.5,4.5))

        _min = 1e6
        _max = 0.0

        _size = self.z.size
        for ii in range(_size):

            _sol = self.solutions[ii]
            _z = self.z[ii]

            for jj, _x in enumerate(_sol): 
                _I =  _z / np.exp(1/_x**self.gamma)
                if self.slopes[ii][jj] > 0:
                    ax.plot(_z,_I,marker='o',ms=1.5,c='k')
                else:
                    ax.plot(_z,_I,marker='o',ms=1.5,c='m')

                if _I < _min:
                    _min = _I
                if _I > _max:
                    _max = _I

        ax.axvline(self.z_critical,lw=1,c='r')
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1) #,direction='in')
        ax.tick_params(which='major',length=5)
        ax.set_rasterization_zorder = 1000000000

        ax.set_yscale('log')

        ax.axis([self.z.min(),self.z.max(),0,_max*1.05])

        ax.annotate(r'z$_{critical}$',xy=(self.z_critical*1.05,_max*1.1),
                    xycoords='data',c='r',annotation_clip=False)
        
        ax.annotate(r'$\gamma$'+f'={self.gamma}, '+r'$\beta$'+f'={self.beta}',
                    xy=(0.7,0.2),xycoords='axes fraction',c='r',annotation_clip=False)
        
        _frac = self.y / self.y_critical
        ax.annotate(f'y={_frac:.3f}'+r'$\times$y$_0$',xy=(0.7,0.15),xycoords='axes fraction',
                    c='r',annotation_clip=False)

        ax.set_xlabel(r'z(V)')
        ax.set_ylabel(r'current')

        plt.savefig(f'bistability_IV_{self.gamma}_beta_{self.beta}_y0_{_frac:.3f}.png',
                    dpi=300, bbox_inches='tight')
        
        plt.show()

    # ----------------------------------------------------------------------------------------------

    def plot_J(self):

        print(self.y_critical)

        if self.rank != 0:
                    return
        
        fig, ax = plt.subplots(figsize=(4.5,4.5))

        _b = self.beta
        _g = self.gamma
        z = 2*self.z_critical
        y = 0.6*self.y_critical
        x = np.linspace(y,0.5,100000)
        _L = self._evaluate_Lx(x=x,y=y,z=z)
        ax.plot(x,_L,lw=1.5,c='k')
        ax.axhline(0,lw=1,c='k',ls='--')

        # x = np.linspace(0,0.5,100000)
        # _L = -(z**2 * np.exp(-1/x**_g) + y**_b - x**_b)
        # ax.plot(x,_L,lw=1,c='k',ls=(0,(2,1,1,1)))

        _zeros, _diff = self._get_zeros(_L)
        _x0 = x[_zeros]

        for _xx in _x0:
            ax.plot(_xx,0,marker='o',ms=4,c='b')
        
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.minorticks_on()
        ax.tick_params(which='both',width=1) #,direction='in')
        ax.tick_params(which='major',length=5)
        ax.set_rasterization_zorder = 1000000000

        ax.axis([0,0.5,-1e-8,1e-7])

        # ax.annotate(r'z$_{critical}$',xy=(self.z_critical*1.05,_max*1.1),
        #             xycoords='data',c='r',annotation_clip=False)
        
        ax.annotate(r'$\gamma$'+f'={self.gamma}, '+r'$\beta$'+f'={self.beta}',
                    xy=(0.25,0.9),xycoords='axes fraction',c='r',annotation_clip=False)
        ax.annotate(r'y=0.6$\times$y$_0$',xy=(0.25,0.85),xycoords='axes fraction',
                    c='r',annotation_clip=False)
        ax.annotate(r'z=2.0$\times$z$_0$',xy=(0.25,0.8),xycoords='axes fraction',
                    c='r',annotation_clip=False)

        ax.set_xlabel(r'x(T$_e$)')
        ax.set_ylabel(r'L(x)')

        plt.savefig(f'Udot.png',dpi=300, bbox_inches='tight')
        
        # plt.show()

    # ----------------------------------------------------------------------------------------------


        

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    bistable = c_bistable(gamma=1,beta=6)
    
    # bistable.get_bistability_region(nx=10000,ny=1000,nz=1000)
    # bistable.plot_bistability_region()
    
    # frac = [0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.04, 1.05, 1.1, 1.5, 2.0, 10.0]
    # for _f in frac:
    #     bistable.solve_bistability_fixed_z(frac=_f,nx=100000,ny=500)
    #     bistable.plot_solutions_fixed_z()

    # frac = [0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.04, 1.05, 1.1, 1.5, 2.0, 10.0]
    # for _f in frac:
    #     bistable.solve_bistability_fixed_y(frac=_f,nx=100000,nz=500)
    #     bistable.plot_solutions_fixed_y()

    # frac = [0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.04, 1.05, 1.1, 1.5, 2.0, 10.0]
    # for _f in frac:
    #     bistable.solve_bistability_fixed_y(frac=_f,nx=100000,nz=500)
    #     bistable.plot_IV_fixed_y()

    # z0 = bistable.z_critical
    # bistable.plot_J()
    
    # frac = [0.3, 0.35, 0.4, 0.45, 0.5]
    # for _f in frac:
    #     bistable.solve_bistability_fixed_y(frac=_f,nx=100000,nz=1000,zhi=6)
    #     bistable.plot_IV_fixed_y()


    nx = 1000
    x = np.linspace(0,2,nx)
    arr = np.exp(-1/x)
    alg = x**4

    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.plot(x,arr,c='b',lw=1.5,label='Arrhenius')
    ax.plot(x,alg,c='r',lw=1.5,label='algebraic')

    # ax.set_yscale('log')

    plt.show()
