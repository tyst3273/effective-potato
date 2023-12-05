
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class octahedra:

    def __init__(self,G=1,g=1,M=4,m=1):
        
        self.g = float(g)
        self.G = float(G)
        self.M = float(M)
        self.m = float(m)

        self.pos = np.array([
                [ 0, 0, 0], # M
                [ 1, 0, 0], # m
                [ 0, 1, 0], # m
                [ 0, 0, 1], # m
                [-1, 0, 0], # m
                [ 0,-1, 0], # m
                [ 0, 0,-1]], # m
                dtype=float)
        self.natoms = self.pos.shape[0]
        self.types = np.array([1,2,2,2,2,2,2],dtype=int)

    def get_fc(self,k,d,l):
        
        """
        Phi^mu,nu_i,j = -K * (r^mu_i - r^mu_j) * (r^nu_i - r^nu_j) / l^2 
        """

        fc = np.zeros((3,3),dtype=float)
        for ii in range(3): # mu
            for jj in range(3): # nu
                fc[ii,jj] = d[ii]*d[jj]
        return -k*fc/l**2

    def get_fc_array(self):
        
        _fc = np.zeros((self.natoms,self.natoms,3,3))

        for ii in range(self.natoms):

            _tii = self.types[ii]
            _pii = self.pos[ii]

            for jj in range(self.natoms):

                if ii == jj:
                    continue

                _tjj = self.types[jj]
                _pjj = self.pos[jj]

                d = _pii-_pjj

                # check if further apart than nn 
                if np.linalg.norm(d) > np.sqrt(2)+1e-3:
                    continue

                if _tii == _tjj: # vertex to vertex
                    k = self.g
                    l = np.sqrt(2)
                else: # vertex to center
                    k = self.G
                    l = 1.0

                _fc[ii,jj] += self.get_fc(k,d,l)

        # self terms ...
        for ii in range(self.natoms):
            _fc[ii,ii,...] = -_fc[ii,...].sum(axis=0)

        # reshape
        self.fc = np.zeros((self.natoms*3,self.natoms*3),dtype=float)
        for ii in range(self.natoms):
            _tii = self.types[ii]

            for jj in range(self.natoms):
                _tjj = self.types[jj]

                if _tii == 1 and _tjj == 1: # vertex to vertex
                    _m = self.M
                elif _tii != _tjj: # vertex to center
                    _m = np.sqrt(self.m*self.M)
                else:
                    _m = self.m

                self.fc[ii*3:(ii+1)*3,jj*3:(jj+1)*3] = _fc[ii,jj,...]/_m

        np.savetxt('fc',self.fc,fmt='% 4.2f')

    def solve(self):
        
        _evals, _evecs = eigh(self.fc,
                check_finite=False,driver='evr',lower=False)

        _evals = np.round(_evals,6)
        #_evecs = np.round(_evecs,6) # [ atom*xyz, mode ]

        _neg = -1*(_evals < 0.0).astype(float) + (_evals >= 0.0).astype(float)
        freq = np.sqrt(np.abs(_evals))*_neg

        print(freq)

        self.evecs = np.zeros((self.natoms*3,self.natoms,3)) # [modes, atoms, xyz]
        for ii in range(self.natoms*3):
            self.evecs[ii,...] = _evecs[:,ii].reshape(self.natoms,3)

        print(_evecs[:,0])
        print(self.evecs[0,...])

        self.disp = np.copy(self.evecs)
        for ii in range(self.natoms):

            _t = self.types[ii]
            if _t == 1:
                _m = self.M
            else:
                _m = self.m

            self.disp[:,ii,:] /= np.sqrt(_m)

    def plot_displacement(self,mode=0):
        
        c = np.zeros((self.natoms,3),dtype=float)
        s = np.zeros(self.natoms)

        _inds = np.flatnonzero(self.types == 1)
        c[_inds,2] = 1.0
        s[_inds] = 100

        _inds = np.flatnonzero(self.types == 2)
        c[_inds,0] = 1.0
        s[_inds] = 40

        x = self.pos[:,0]; y = self.pos[:,1]; z = self.pos[:,2]
        dx = self.disp[mode,:,0]; dy = self.disp[mode,:,1]; dz = self.disp[mode,:,2]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d',aspect='equal')

        ax.scatter(x, y, z, marker='o', s=s, c=c)

        ax.quiver(x,y,z,dx,dy,dz)
        plt.show()


if __name__ == '__main__':

    o = octahedra()
    o.get_fc_array()
    o.solve()

    o.plot_displacement(mode=7)


