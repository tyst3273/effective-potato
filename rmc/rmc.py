
import numpy as np
import matplotlib.pyplot as plt

class model:
    
    def __init__(self,n_sites=1000,a=1):
        self.n_sites = n_sites 
        self.a = a
        self.pos = np.arange(-n_sites//2,n_sites//2)*self.a
        self.vol = self.n_sites*self.a
        self.inds = np.arange(self.n_sites)
        self.disp = np.zeros(n_sites)

    def set_exponential_corr(self,corr_len=1,amplitude=1):
        self.exp_corr = np.exp(-np.abs(self.pos)/corr_len)*amplitude

    def set_gaussian_corr(self,fwhm=50,amplitude=10):
        c = fwhm/np.sqrt(8*np.log(2))
        self.exp_corr = np.exp(-0.5*(self.pos/c)**2)*amplitude

    def calculate_disp_corr(self):
        _disp = self.disp
        _ft = np.fft.fft(_disp,norm='ortho')
        self.corr = np.fft.ifft(np.abs(_ft)**2,norm='ortho')/np.sqrt(self.vol)
        self.corr = np.real(self.corr)
        self.corr = np.fft.fftshift(self.corr)

    def do_rmc(self,beta=0.5,tol=1e-3,max_iter=1000,step_size=0.5):

        self.error_sq = np.zeros(max_iter)
        self.delta_sq = np.zeros(max_iter)

        self.beta = beta
        self.step_size = step_size
        self.chi_sq = 0 # have to initialize this to 0

        self.calculate_disp_corr()
        self._calc_chi_squared()

        converged = False
        for ii in range(max_iter):

            if ii % 100 == 0:
               print(f'chi_sq[{ii}]:',self.chi_sq)
               print(f'delta_chi_sq[{ii}]:',self.delta_chi_sq)

            if np.abs(self.delta_chi_sq) <= tol:
                converged = True
                break
            
            self._move()
            self.calculate_disp_corr()

            keep = self._check_move()

            self.error_sq[ii] = self.chi_sq
            self.delta_sq[ii] = self.delta_chi_sq

            if keep:
                #print('keep!')
                continue
            else: 
                #print('reject!')
                self._unmove()

        if converged:
            print(f'converged after {ii+1} steps!')
        else:
            print(f'failed to converged after {ii+1} steps!')
        print('final chi squared:',self.chi_sq)
        print('final delta chi squared:',self.delta_chi_sq)

    def _move(self,step_size=0.5,num_moves=1):
        np.random.shuffle(self.inds)
        self.move_inds = self.inds[:50]
        self.move_steps = np.random.standard_normal(size=num_moves)*step_size
        self.disp[self.move_inds] += self.move_steps

    def _unmove(self):
        self.disp[self.move_inds] -= self.move_steps

    def _check_move(self):
        keep = False
        self._calc_chi_squared()
        _weight = np.exp(-self.delta_chi_sq*self.beta/2)
        _cutoff = min(1.0,_weight)
        _random = np.random.uniform()
        if _random < _cutoff:
            keep = True
        return keep

    def _calc_chi_squared(self):
        self.last_chi_sq = np.copy(self.chi_sq)
        self.chi_sq = np.sum((self.exp_corr-self.corr)**2)
        self.delta_chi_sq = self.chi_sq-self.last_chi_sq

    def plot(self):
        plt.plot(self.pos,self.exp_corr,c='r',label='experiment')
        plt.plot(self.pos,self.corr,c='b',label='calc')
        plt.legend()
        plt.show()

        plt.clf()

        plt.plot(np.arange(self.error_sq.size),self.error_sq,lw=0,
            marker='o',c='r',label='error_sq')
        plt.plot(np.arange(self.delta_sq.size),self.delta_sq,lw=0,
            marker='^',c='b',label='delta_sq')
        plt.legend()
        plt.show()

        plt.clf()

        plt.scatter(self.pos,self.disp)
        plt.show()

    def _calc_explicit_disp_corr(self):

        self.explicit_corr = np.zeros(self.n_sites)
        for ii in range(self.n_sites):
            self.explicit_corr[ii] = np.trapz(_disp*np.roll(_disp,ii),self.pos)
        self.explicit_corr = self.explicit_corr/self.vol

        plt.plot(self.pos,self.explicit_corr,c='r',label='explicit')
        plt.plot(self.pos,self.corr,c='b',label='ft')
        plt.legend()
        plt.show()





if __name__ == '__main__':

    model = model()

    model.set_exponential_corr(corr_len=15,amplitude=10)
    #model.set_gaussian_corr()

    model.do_rmc(beta=10,tol=1e-6,max_iter=25000,step_size=0.005)
    model.plot()








