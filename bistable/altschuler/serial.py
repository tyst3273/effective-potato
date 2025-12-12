import numpy as np
import matplotlib.pyplot as plt

beta = 6
gamma = 1

y0 = ( beta/gamma + 1 ) ** (- ( beta + gamma )/( beta*gamma ) )
z0 = np.sqrt( beta/gamma * ( gamma * np.e / ( gamma + beta ) ) ** ( ( gamma + beta ) / gamma ) )

ny = 500
nz = 1000
y = np.linspace(0.01,y0*1.1,ny)
z = np.linspace(0.01,z0*3,nz)

bistable = np.zeros((nz,ny),dtype=int)

for ii, zz in enumerate(z):
    print(ii)

    for jj, yy in enumerate(y):

        x = np.linspace(yy,1,100)

        F = x ** beta - yy ** beta - zz ** 2 * np.exp( - 1 / x ** gamma )
        zeros = np.flatnonzero(np.diff(np.sign(F)))

        if zeros.size > 1:
            bistable[ii,jj] = 1

fig, ax = plt.subplots(figsize=(4.5,4.5))

ax.imshow(bistable, origin='lower', extent=(y.min(), y.max(), z.min(), z.max()), 
          aspect='auto', cmap='binary', vmin=0, vmax=1, interpolation='none')

ax.axhline(z0,lw=1,color='r')

ax.axvline(y0,lw=1,c='r')


ax.set_xlabel('y')
ax.set_ylabel('z')
ax.set_title('Bistability Region')
plt.show()

