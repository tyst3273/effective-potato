import numpy as np
import matplotlib.pyplot as plt 

gamma = 5.0

y = 0.1
z = np.linspace(0,gamma,1000)

fig, ax = plt.subplots(1,2,figsize=(8,3))

f = gamma * np.exp(-1/(y+z))

ax[0].plot(z,f-z,c='m',label='$L(z)$')
ax[0].axhline(0,lw=1,ls='--',c='k')
ax[0].legend(loc='upper right',frameon=False)

axi = ax[0].inset_axes(
    [0.05,0.05,0.45,0.45], xlim=(0, 0.001), ylim=(-0.0005, 0.0005), 
    xticks=[], yticks=[])
axi.plot(z,f-z,c='m',label='$L(z)$')
axi.axhline(0,lw=1,ls='--',c='k')
ax[0].indicate_inset_zoom(axi, edgecolor='k')

ax[1].plot(z,f,c='b',label='$\lambda e^{1/(y+z)}$')
ax[1].plot(z,z,c='r',label='z')
ax[1].legend(frameon=False)

ax[0].set_xlabel('z')
ax[1].set_xlabel('z')

# ax[1].set_yscale('log')

ax[0].axis([0,gamma,-2,1.5])
ax[1].axis([0,gamma,0,4.5])

fig.suptitle('$\lambda=5.0,~y=0.1$')

plt.savefig('example_dot_n.png',dpi=200,format='png',bbox_inches='tight')
plt.show()
