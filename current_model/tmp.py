import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45

k_B = 8.6173303e-2 # meV/K
C2K = 273

# --------------------------------------------------------------------------------------------------

class c_solver:
    
    # ----------------------------------------------------------------------------------------------
    
    def __init__(self,E_0,j_0,rho_0,epsilon,c_p,alpha,T_c):
        
        """
        power = V/cm/(ohm*cm) => W/cm^3
        
        example resistivities:
            metals ~ 1e-8 Ohm*cm
            germanium ~= 4.6 Ohm*cm
            GaAs ~ 1e-3 - 1e8 Ohm*cm 
            silicon ~=  2.3e4 Ohm*cm
            glass ~ 1e11 - 1e15 Ohm*cm
            air ~ 1e9 - 1e15 Ohm*cm
            
        """
        
        self.E_0 = E_0 # V/cm
        self.j_0 = j_0 # mA/mm^2
        self.j_0 *= 0.1 # mA/mm^2 = 0.1 A/cm^2 
        self.rho_0 = rho_0 # Ohm*cm 
        self.epsilon = epsilon # meV
        self.c_p = c_p # J/cm^3/K
        self.alpha = alpha # W/cm^3/K
        self.T_c = T_c # C
        self.T_c += C2K # K = C + 273 
        
        self.pE_0 = self.E_0**2 / self.rho_0
        self.pj_0 = self.j_0**2 * self.rho_0
        
        # print(self.pE_0)
        # print(self.pj_0)
        
    # ----------------------------------------------------------------------------------------------
    
    def _constant_E_diffy_q(self,t,T):
        
        """
        dT/dt = p^E_0/c_p * ( exp( -e/kT ) - gamma_E * (T-T_0) )
        """
        
        _a = self.pE_0 / self.c_p
        _b = -self.epsilon / k_B 
        _c = self.alpha / self.pE_0
        _T_c = self.T_c
        
        # print(_a,_b,_c,_T_c)
        
        return _a * ( np.exp(_b/T) - _c * (T-_T_c) )
    
    # ----------------------------------------------------------------------------------------------
    
    def _constant_j_diffy_q(self,t,T):
        
        """
        dT/dt = p^j_0/c_p * ( exp( e/kT ) - gamma_j * (T-T_0) )
        """
        
        _a = self.pj_0 / self.c_p
        _b = -self.epsilon / k_B 
        _c = self.alpha / self.pj_0
        _T_c = self.T_c
        
        return _a * ( np.exp(_b/T) - _c * (T-_T_c) )    
    
    # ----------------------------------------------------------------------------------------------
    
    def _calculate_resistivity(self,T):
        
        """
        rho(T) = rho_0 exp( e /kT )
        """
        return self.rho_0 * np.exp( self.epsilon/k_B/T )
     
    # ----------------------------------------------------------------------------------------------
    
    def solve_constant_E(self,num_steps=1000,t0=None,T0=None,j_cut=None):
        
        """
        """
        
        if T0 is None:
            T0 = np.array([self.T_c],dtype=float)
        else:
            T0 = np.array([T0+C2K],dtype=float)
            
        if t0 is None:
            t0 = 0    
            
        t_bound = 1000
        
        _solver = RK45(self._constant_E_diffy_q,t0,T0,t_bound,
                        atol=1e-12,rtol=1e-9) 
        
        t = np.zeros(num_steps,dtype=float)
        T = np.zeros(num_steps,dtype=float)
        
        _solved = False
        for ii in range(num_steps):

            _solver.step()

            t[ii] = _solver.t
            T_ii = _solver.y[0]
            T[ii] = T_ii
            
            _rho = self._calculate_resistivity(T_ii)
            _j = self.E_0/_rho
            
            # break loop if current density reaches cutoff
            if j_cut is not None:
                print(_j,j_cut)
                if _j >= j_cut:
                    _last_step = ii
                    _solved = True
                    print(f'\ncurrent density reached cutoff at step {ii+1}\n')
                    break

            # break loop if converged
            if _solver.status == 'finished':
                _last_step = ii
                _solved = True
                print(f'\nfinished at step {ii+1}\n')
                break
            
        if not _solved:
            print(f'\n*** fuck! ***\nfailed to solve after {ii+1} steps!\n')
            raise KeyboardInterrupt()
                    
        t = t[:_last_step+1]
        T = T[:_last_step+1]

        _rho = self._calculate_resistivity(T)
        _j = self.E_0/_rho
        plt.plot(t,1/_rho,c='k')
        plt.show()
        
        return t, T
        
    # ----------------------------------------------------------------------------------------------
    
    def solve_constant_j(self,num_steps=1000,t0=None,T0=None):
        
        """
        """
        
        if T0 is None:
            T0 = np.array([self.T_c],dtype=float)
        else:
            T0 = np.array([T0+C2K],dtype=float)
            
        if t0 is None:
            t0 = 0    
            
        t_bound = 1000
        
        _solver = RK45(self._constant_j_diffy_q,t0,T0,t_bound,
                        atol=1e-12,rtol=1e-9) 
        
        t = np.zeros(num_steps,dtype=float)
        T = np.zeros(num_steps,dtype=float)
        
        _solved = False
        for ii in range(num_steps):

            _solver.step()

            t[ii] = _solver.t
            T_ii = _solver.y[0]
            T[ii] = T_ii

            # break loop if converged
            if _solver.status == 'finished':
                _last_step = ii
                _solved = True
                print(f'\nfinished at step {ii+1}\n')
                break
            
        if not _solved:
            print(f'\n*** fuck! ***\nfailed to solve after {ii+1} steps!\n')
            raise KeyboardInterrupt()
                    
        t = t[:_last_step+1]
        T = T[:_last_step+1]

        # _rho = self._calculate_resistivity(T)
        # _j = self.E_0/_rho
        
        plt.plot(t,T,c='k')
        plt.show()
        
        return t, T
    
    # ----------------------------------------------------------------------------------------------
    

if __name__ == '__main__':    
    
    E_0 = 50 # V/cm
    j_0 = 100 # mA/mm^2
    rho_0 = 1e2 # Ohm*cm ; typical semiconductors 10^8-10^18 Ohm * cm
    epsilon = 15 # meV
    c_p = 4 # J/cm^3/K
    alpha = 1 # W/cm^3/K
    T_c = 600 # C
    
    j_cut = 4.5
    
    solver = c_solver(E_0,j_0,rho_0,epsilon,c_p,alpha,T_c)        
    t, T = solver.solve_constant_E(T0=1000)#j_cut=j_cut)
    
    
    #solver.solve_constant_j(T0=T[-1])
    
    
    # solver.solve_constant_j()


    
