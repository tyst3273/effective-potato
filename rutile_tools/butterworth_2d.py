

import numpy as np
import matplotlib.pyplot as plt


def butterworth_2d(x,y,wx=1,wy=1,cx=0,cy=0,n=40):
    f = 1/(1+((x-cx)/(wx/2))**n)/(1+((y-cy)/(wy/2))**n)
    return f

w = 1

x_max = 1
nx = 101
x = np.linspace(-x_max,x_max,nx)
x, y = np.meshgrid(x,x,indexing='ij')

extent = [-x_max,x_max,-x_max,x_max]

fig, ax = plt.subplots(1,3,figsize=(12,4),gridspec_kw={'wspace':0.1,'hspace':0.1})

b1 = butterworth_2d(x,y,wx=w,wy=w,cx=0,cy=0,n=20)
ax[0].imshow(b1.T,aspect='auto',origin='lower',cmap='Purples',extent=extent,vmin=0,vmax=1)

b2 = butterworth_2d(x,y,wx=w,wy=w/2,cx=0,cy=0.25,n=20)
ax[1].imshow(b2.T,aspect='auto',origin='lower',cmap='Purples',extent=extent,vmin=0,vmax=1)

a = np.array([[1,0],[0,1]],dtype=float)
b = np.array([[1,1],[-1,1]],dtype=float).T/np.sqrt(2)
R = np.linalg.solve(a.T,b.T)

xp = R[0,0]*x+R[0,1]*y
yp = R[1,0]*x+R[1,1]*y

b3 = butterworth_2d(xp,yp,wx=w,wy=w/2,cx=0,cy=0,n=20)
ax[2].imshow(b3.T,aspect='auto',origin='lower',cmap='Purples',extent=extent,vmin=0,vmax=1)


# b2 = butterworth_1d(x,w=w,c=0,n=10)
# b3 = butterworth_1d(x,w=w,c=0,n=50)

# ax.plot([-x_max,-w/2],[0,0],lw=2,ls='-',c='k')
# ax.plot([-w/2,-w/2],[0,1],lw=2,ls='-',c='k')
# ax.plot([w/2,w/2],[0,1],lw=2,ls='-',c='k')
# ax.plot([-w/2,w/2],[1,1],lw=2,ls='-',c='k')
# ax.plot([w/2,x_max],[0,0],lw=2,ls='-',c='k')

# ax.plot([0,0],[-1,1.5],lw=1,ls=(0,(2,2)),c='k')

# ax.plot(x,b1,c='m',ls=(0,(2,2)),lw=1,label='n=4')
# ax.plot(x,b2,c='m',ls=(0,(4,2)),lw=1,label='n=10')
# ax.plot(x,b3,c='m',ls='-',lw=1,label='n=50')

# ax.legend(frameon=False,fontsize='x-large',loc='upper right',handlelength=1.5,handletextpad=0.5)


           
# format plots
for ii in range(3):
    for axis in ['top','bottom','left','right']:
        ax[ii].spines[axis].set_linewidth(1.5)
    ax[ii].minorticks_on()
    ax[ii].tick_params(which='both',width=1,labelsize='x-large')
    ax[ii].tick_params(which='major',length=5)
    ax[ii].tick_params(which='minor',length=2)
    # ax[ii].set_xticks([-1,-0.5,0,0.5,1])
    # ax.set_yticks(np.arange(-0.4,0.6,0.2))

    ax[ii].axis(extent)

# ax[0,1].legend(frameon=False,fontsize='x-large')

# ax.set_xlabel('x',labelpad=4,fontsize='x-large')
# ax.set_ylabel('weight',labelpad=4,fontsize='x-large')

# ax[0,0].annotate('a)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[0,1].annotate('b)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[1,0].annotate('c)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[1,1].annotate('d)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
        
plt.savefig('butterworth_2d.pdf',dpi=150,bbox_inches='tight')

# plt.show()

