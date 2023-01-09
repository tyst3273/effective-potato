

import numpy as np
import os


def get_nn(ii,pos,lx=5,ly=5):

    pii = pos[ii,:]
    pii = np.tile(pii.reshape(1,2),reps=(pos.shape[0],1))
    rel = pos-pii

    dr = -(rel[:,0] > lx/2).astype(int)
    dr += (rel[:,0] <= -lx/2).astype(int)
    rel[:,0] += lx*dr

    dr = -(rel[:,1] > ly/2).astype(int)
    dr += (rel[:,1] <= -ly/2).astype(int)
    rel[:,1] += ly*dr

    dist = np.sqrt(np.sum(rel**2,axis=1))

    return rel, dist


lx = 5; ly = 5
cut = 1.0
d = 0.0001

pos = np.loadtxt('pos.dat',skiprows=9)
types = pos[:,1].astype(int)
pos = pos[:,2:4]

atoms = [0,1,2]
for aa in atoms:

    vecs, lens = get_nn(aa,pos)
    
    fxp = np.loadtxt(f'{aa}+x/force.dat',skiprows=9)[:,2:4]
    fxm = np.loadtxt(f'{aa}-x/force.dat',skiprows=9)[:,2:4]
    fcx = -(fxp-fxm)/(2*d) # negative sign since FC is derivative of -F

    fyp = np.loadtxt(f'{aa}+y/force.dat',skiprows=9)[:,2:4]
    fym = np.loadtxt(f'{aa}-y/force.dat',skiprows=9)[:,2:4]
    fcy = -(fyp-fym)/(2*d)

    inds = np.flatnonzero(lens <= cut)
    vecs = vecs[inds]
    fcx = fcx[inds,:]
    fcy = fcy[inds,:]
    
    t1 = np.tile(types[aa],inds.size)
    t2 = types[inds]

    p = np.c_[t1,t2,vecs,fcx,fcy]

    np.savetxt(f'{aa}.dat',p,fmt='%2g %2g % 5.3f % 5.3f % 12.7f % 12.7f % 12.7f % 12.7f')






