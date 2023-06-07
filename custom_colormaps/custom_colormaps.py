
import numpy as np
from matplotlib.colors import ListedColormap

# --------------------------------------------------------------------------------------------------

def make_colormap(colors,positions):

    nc = 256
    arr = np.ones((nc,4),dtype=float)
    positions = np.array(positions,dtype=float)
    positions *= nc/positions.max()
    positions = positions.astype(int)

    for ii in range(len(colors)-1):

        c0 = colors[ii]
        if len(c0) == 3:
            c0.append(1.0)

        c1 = colors[ii+1]
        if len(c1) == 3:
            c1.append(1.0)

        for jj in range(4):
            arr[positions[ii]:positions[ii+1],jj] = \
                np.linspace(c0[jj],c1[jj],positions[ii+1]-positions[ii])

    arr[:,:3] = arr[:,:3]/255
    cmap = ListedColormap(arr)
    
    return cmap

# --------------------------------------------------------------------------------------------------
# custom MATLAB 'parula' cmap

colors = [[89,0,174],[103,0,209],[112,0,235],[117,0,252],[115,36,255],[103,80,255],[92,114,248],
          [77,139,235],[61,159,224],[26,175,207],[0,188,183],[0,199,157],[14,208,125],
          [85,212,84],[139,211,36],[188,204,0],[225,199,0],[248,202,34],[241,224,0],
          [232,246,0],[233,255,0]]
positions = np.arange(len(colors))
parula_cmap = make_colormap(colors,positions)

# --------------------------------------------------------------------------------------------------


