
import numpy as np

class octahedra:

    def __init__(self,G=1,g=1,M=1,m=1):
        
        self.g = float(g)
        self.G = float(G)
        self.M = float(m)
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
        Phi^mu,nu_i,j = -K * (r^mu_i - r^mu_j) * (r^nu_i - r^nu_k) / l^2 
        """

        fc = np.zeros((3,3),dtype=float)
        for ii in range(3): # mu
            for jj in range(3): # nu
                fc[ii,jj] = d[ii]*d[jj]
        return -k*fc/l**2

    def get_fc_array(self):

        self.fc = np.zeros((self.natom,self.natom,3,3))

        for ii in range(self.natom):

            _tii = self.types[ii]
            _pii = self.pos[ii]

            for jj in range(self.natom):

                if ii == jj:
                    continue

                _tjj = self.types[jj]
                _pjj = self.pos[jj]

                # check if further apart than nn 
                if np.linalg.norm(_pii-_pjj) > np.sqrt(2)+1e-3:
                    continue

                if _tii == _tjj: # vertex to vertex
                    k = self.g
                    l = np.sqrt(2)
                else: # vertex to center
                    k = self.G
                    l = 1.0

                d = _pii-_pjj

                _fc = self.get_fc(k,d,l)
                self.fc[ii,jj] = _fc
            
        # self terms ...
#        for ii in range(self.natom):
#            self.fc[ii,ii




if __name__ == '__main__':

    o = octahedra()
    o.get_fc_array()

