

import numpy as np
import matplotlib.pyplot as plt


def butterworth_1d(x,w=1,c=0,n=40):
    f = 1/(1+((x-c)/(w/2))**n)
    return f


w = 1

x_max = 1
nx = 1001
x = np.linspace(-x_max,x_max,nx)


fig, ax = plt.subplots(figsize=(6,5))

b1 = butterworth_1d(x,w=w,c=0,n=4)
b2 = butterworth_1d(x,w=w,c=0,n=10)
b3 = butterworth_1d(x,w=w,c=0,n=50)

ax.plot([-x_max,-w/2],[0,0],lw=2,ls='-',c='k')
ax.plot([-w/2,-w/2],[0,1],lw=2,ls='-',c='k')
ax.plot([w/2,w/2],[0,1],lw=2,ls='-',c='k')
ax.plot([-w/2,w/2],[1,1],lw=2,ls='-',c='k')
ax.plot([w/2,x_max],[0,0],lw=2,ls='-',c='k')

ax.plot([0,0],[-1,1.5],lw=1,ls=(0,(2,2)),c='k')

ax.plot(x,b1,c='m',ls=(0,(2,2)),lw=1,label='n=4')
ax.plot(x,b2,c='m',ls=(0,(4,2)),lw=1,label='n=10')
ax.plot(x,b3,c='m',ls='-',lw=1,label='n=50')

ax.legend(frameon=False,fontsize='x-large',loc='upper right',handlelength=1.5,handletextpad=0.5)

           
# format plots
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
ax.minorticks_on()
ax.tick_params(which='both',width=1,labelsize='x-large')
ax.tick_params(which='major',length=5)
ax.tick_params(which='minor',length=2)
ax.set_xticks([-1,-0.5,0,0.5,1])
# ax.set_yticks(np.arange(-0.4,0.6,0.2))

ax.axis([-1,1,-0.05,1.05])


# ax[0,1].legend(frameon=False,fontsize='x-large')

ax.set_xlabel('x',labelpad=4,fontsize='x-large')
ax.set_ylabel('weight',labelpad=4,fontsize='x-large')

# ax[0,0].annotate('a)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[0,1].annotate('b)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[1,0].annotate('c)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
# ax[1,1].annotate('d)',xy=(-1.5,1.5e4),xycoords='data',fontsize='x-large')
        
plt.savefig('butterworth.pdf',dpi=150,bbox_inches='tight')

# plt.show()

