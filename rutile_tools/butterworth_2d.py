

import numpy as np
import matplotlib.pyplot as plt


def butterworth_2d(x,y,wx=1,wy=1,cx=0,cy=0,n=40):
    f = 1/(1+((x-cx)/(wx/2))**n)/(1+((y-cy)/(wy/2))**n)
    return f

n = 8

w = 4
wy = w/2
cx = 1
cy = 1

x_max = 4
nx = 25
xx = np.linspace(-x_max,x_max,nx)
x, y = np.meshgrid(xx,xx,indexing='ij')

extent = [-x_max,x_max,-x_max,x_max]

fig, ax = plt.subplots(1,3,figsize=(12,3.3),gridspec_kw={'wspace':0.1,'hspace':0.1})

b1 = butterworth_2d(x,y,wx=w,wy=w,cx=0,cy=0,n=n)
ax[0].imshow(b1.T,aspect='auto',origin='lower',cmap='Purples',extent=extent,vmin=0,vmax=1)

b2 = butterworth_2d(x,y,wx=w,wy=wy,cx=cx,cy=cy,n=n)
ax[1].imshow(b2.T,aspect='auto',origin='lower',cmap='Purples',extent=extent,vmin=0,vmax=1)

a = np.array([[1,0],[0,1]],dtype=float)
b = np.array([[1,1],[-1,1]],dtype=float).T/np.sqrt(2)
R = np.linalg.solve(a.T,b.T)

xp = R[0,0]*x+R[0,1]*y
yp = R[1,0]*x+R[1,1]*y

cp = R@np.array([cx,cy])

b3 = butterworth_2d(xp,yp,wx=w,wy=wy,cx=cp[0],cy=cp[1],n=n)
im = ax[2].imshow(b3.T,aspect='auto',origin='lower',cmap='Purples',extent=extent,vmin=0,vmax=1)

fig.colorbar(im,ax=[ax[0],ax[1],ax[2]],extend='both',aspect=30,pad=0.015)

ax[0].plot([-w/2,w/2],[-w/2,-w/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)
ax[0].plot([-w/2,w/2],[w/2,w/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)
ax[0].plot([w/2,w/2],[-w/2,w/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)
ax[0].plot([-w/2,-w/2],[-w/2,w/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)

ax[1].plot([cx-w/2,cx+w/2],[cy-wy/2,cy-wy/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)
ax[1].plot([cx-w/2,cx+w/2],[cy+wy/2,cy+wy/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)
ax[1].plot([cx+w/2,cx+w/2],[cy-wy/2,cy+wy/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)
ax[1].plot([cx-w/2,cx-w/2],[cy-wy/2,cy+wy/2],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)

a = np.array([cx-w/2-cx,cy-wy/2-cy]); b = np.array([cx+w/2-cx,cy-wy/2-cy])
a = R.T@a; b = R.T@b
ax[2].plot([a[0]+cx,b[0]+cx],[a[1]+cy,b[1]+cy],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)

a = np.array([cx-w/2-cx,cy+wy/2-cy]); b = np.array([cx+w/2-cx,cx+wy/2-cy])
a = R.T@a; b = R.T@b
ax[2].plot([a[0]+cx,b[0]+cx],[a[1]+cy,b[1]+cy],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)

a = np.array([cx+w/2-cx,cy-wy/2-cy]); b = np.array([cx+w/2-cx,cx+wy/2-cy])
a = R.T@a; b = R.T@b
ax[2].plot([a[0]+cx,b[0]+cx],[a[1]+cy,b[1]+cy],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)

a = np.array([cx-w/2-cx,cy-wy/2-cy]); b = np.array([cx-w/2-cx,cx+wy/2-cy])
a = R.T@a; b = R.T@b
ax[2].plot([a[0]+cx,b[0]+cx],[a[1]+cy,b[1]+cy],ls=(0,(2,2)),c='k',lw=1.5,alpha=1)

xx = np.linspace(-x_max,x_max,nx+1)
for ii in range(nx):
    _x = xx[ii]
    ax[0].plot([-x_max,x_max],[_x,_x],c='g',lw=1,ls='-',alpha=0.25)
    ax[0].plot([_x,_x],[-x_max,x_max],c='g',lw=1,ls='-',alpha=0.25)
    ax[1].plot([-x_max,x_max],[_x,_x],c='g',lw=1,ls='-',alpha=0.25)
    ax[1].plot([_x,_x],[-x_max,x_max],c='g',lw=1,ls='-',alpha=0.25)
    ax[2].plot([-x_max,x_max],[_x,_x],c='g',lw=1,ls='-',alpha=0.25)
    ax[2].plot([_x,_x],[-x_max,x_max],c='g',lw=1,ls='-',alpha=0.25)
    
ax[0].plot(0,0,c='r',marker='x',ms=8)
ax[1].plot(cx,cy,c='r',marker='x',ms=8)
ax[2].plot(cx,cy,c='r',marker='x',ms=8)

ax[0].annotate('',xytext=(-3,-3),xy=(-3,-2),xycoords='data',    
               arrowprops={'arrowstyle':'-|>','linewidth':2})
ax[0].annotate('',xytext=(-3,-3),xy=(-2,-3),xycoords='data',    
               arrowprops={'arrowstyle':'-|>','linewidth':2})
ax[0].plot(-3,-3,marker='o',ms=2,c='k')
ax[0].annotate('A',xy=(-3,-3.75),xycoords='data',fontsize='x-large')

ax[1].annotate('',xytext=(-3,-3),xy=(-3,-2),xycoords='data',    
               arrowprops={'arrowstyle':'-|>','linewidth':2})
ax[1].annotate('',xytext=(-3,-3),xy=(-2,-3),xycoords='data',    
               arrowprops={'arrowstyle':'-|>','linewidth':2})
ax[1].plot(-3,-3,marker='o',ms=2,c='k')
ax[1].annotate('A',xy=(-3,-3.75),xycoords='data',fontsize='x-large')

x1 = np.array([1,0],dtype=float)
x2 = np.array([0,1],dtype=float)
x1 = R.T@x1
x2 = R.T@x2

ax[2].annotate('',xytext=(-3,-3),xy=(-3+x1[0],-3+x1[1]),xycoords='data',    
               arrowprops={'arrowstyle':'-|>','linewidth':2})
ax[2].annotate('',xytext=(-3,-3),xy=(x2[0]-3,x2[1]-3),xycoords='data',    
                arrowprops={'arrowstyle':'-|>','linewidth':2})
ax[2].plot(-3,-3,marker='o',ms=2,c='k')
ax[2].annotate('B',xy=(-3.1,-3.75),xycoords='data',fontsize='x-large')

# verts = np.

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
    ax[ii].set_xticks(np.arange(-4,5))
    ax[ii].set_yticks(np.arange(-4,5))
    # ax.set_yticks(np.arange(-0.4,0.6,0.2))
    ax[ii].axis(extent)

# ax[0,1].legend(frameon=False,fontsize='x-large')

ax[0].set_xlabel('x',labelpad=4,fontsize='x-large')
ax[1].set_xlabel('x',labelpad=4,fontsize='x-large')
ax[2].set_xlabel('x',labelpad=4,fontsize='x-large')
ax[0].set_ylabel('y',labelpad=4,fontsize='x-large')

ax[1].set_yticklabels([])
ax[2].set_yticklabels([])

# ax[0,0].annotate('a)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[0,1].annotate('b)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[1,0].annotate('c)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[1,1].annotate('d)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
        
plt.savefig('butterworth_2d.pdf',dpi=150,bbox_inches='tight')

# plt.show()

