import numpy as np
import matplotlib.pyplot as plt 

gamma = 5.0

y = 0.1
x = np.linspace(0,gamma,1000)

fig, ax = plt.subplots(1,2,figsize=(8,3))

f = gamma * np.exp(-1/(x+y))

ax[0].plot(x,f-x,c='m',label='$L(x)$')
ax[0].axhline(0,lw=1,ls='--',c='k')
ax[0].legend(loc='upper right',frameon=False)

ax[1].plot(x,f,c='b',label='$\gamma e^{1/(x+y)}$')
ax[1].plot(x,x,c='r',label='x')
ax[1].legend(frameon=False)

ax[0].set_xlabel('x')
ax[1].set_xlabel('x')

# ax[1].set_yscale('log')

ax[0].axis([0,gamma,-1,1.25])
ax[1].axis([0,gamma,0,4.5])

fig.suptitle('$\gamma=5.0,~y=0.1$')

plt.savefig('example_dot_n.png',dpi=200,format='png',bbox_inches='tight')
plt.show()
