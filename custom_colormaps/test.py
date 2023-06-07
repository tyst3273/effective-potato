
import numpy as np
import matplotlib.pyplot as plt


from custom_colormaps import parula_cmap 


n = 100
x, y = np.meshgrid(np.linspace(0,2*np.pi,n),np.linspace(0,2*np.pi,n),indexing='ij')
f = np.cos(x)*np.cos(y)

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(f,origin='lower',aspect='auto',cmap=parula_cmap,interpolation='None',
    vmin=-0.75,vmax=0.75)
fig.colorbar(im,ax=ax,extend='both')

plt.show()

