
import numpy as np
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
        self.natom = self.pos.shape[0]
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

        _fc = np.zeros((self.natom,self.natom,3,3))

        for ii in range(self.natom):

            _tii = self.types[ii]
            _pii = self.pos[ii]

            for jj in range(self.natom):

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
                    _m = np.sqrt(self.m**2)
                else: # vertex to center
                    k = self.G
                    l = 1.0
                    _m = np.sqrt(self.m*self.M)

                _fc[ii,jj] += self.get_fc(k,d,l)/_m
            
        # self terms ...
        for ii in range(self.natom):
            _fc[ii,ii,...] = -_fc[ii,...].sum(axis=0)

        # reshape
        self.fc = np.zeros((self.natom*3,self.natom*3),dtype=float)
        for ii in range(self.natom):
            for jj in range(self.natom):
                self.fc[ii*3:(ii+1)*3,jj*3:(jj+1)*3] = _fc[ii,jj,...]

        np.savetxt('fc',self.fc,fmt='% 4.2f')

    def solve(self):

        _evals, evecs = eigh(self.fc,
                check_finite=False,driver='evr',lower=False)

        _evals = np.round(_evals,6)
        evecs = np.round(evecs,6)

        _neg = -1*(_evals < 0.0).astype(float) + (_evals >= 0.0).astype(float)
        freq = np.sqrt(np.abs(_evals))*_neg

        print(freq)

        evecs.shape = [self.natom,3]
        print(evecs)








if __name__ == '__main__':

    o = octahedra()
    o.get_fc_array()
    o.solve()


