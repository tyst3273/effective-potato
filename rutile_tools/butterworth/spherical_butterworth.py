

import numpy as np
import matplotlib.pyplot as plt


def butterworth(x,y,z,cx,cy,cz,r,n=40):
    q = np.sqrt((x-cx)**2+(y-cy)**2+(z-cz)**2)
    f = 1/(1+(q/r)**n)
    return f


nx = 401
x = np.linspace(-5,5,nx)
x, y = np.meshgrid(x,x,indexing='ij')
z = 0

w = butterworth(x,y,z,1,1,0,r=1,n=5)
print(w.max())

fig, ax = plt.subplots(figsize=(6.5,6))

extent = [x.min(),x.max(),y.min(),y.max()]
ax.imshow(w.T,extent=extent,aspect='auto',origin='lower',cmap='Blues',vmin=0,vmax=1)

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))

ax.grid(alpha=0.5,c='g')

plt.show()






