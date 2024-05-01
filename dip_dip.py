
import numpy as np
import matplotlib.pyplot as plt

mu0 = 4*np.pi
c = mu0/(4*np.pi)


class c_dip_dip:
    
    def __init__(self,m1=[0,0,1],m2=[0,0,1],r0=[0,0,1],u1=[0,0,0],u2=[0,0,0]):
        
        self.m1 = np.array(m1,dtype=float)
        self.m2 = np.array(m2,dtype=float)
        
        self.r0 = np.array(r0,dtype=float)
        self.r0len = np.linalg.norm(self.r0)
        self.r0hat = self.r0/self.r0len
        
        self.u1 = np.array(u1,dtype=float)
        self.u2 = np.array(u2,dtype=float)
        self.r = self.r0+(self.u2-self.u1)
        self.rlen = np.linalg.norm(self.r)
        self.rhat = self.r/self.rlen
        
        print('m1:',self.m1)        
        print('m2:',self.m2)       
        print('')
        print('r0:',self.r0)    
        print('r0len:',self.r0len)
        print('r0hat',self.r0hat)  
        print('')
        print('u1:',self.u1)   
        print('u2:',self.u2)   
        print('r:',self.r)   
        print('rlen:',self.rlen)
        print('rhat',self.rhat)
        print('')
        
    def calc_energy(self):
        
        _m1 = self.m1; _m2 = self.m2; _rhat = self.rhat; _rlen = self.rlen
        self.energy = -c*(3*(_m1@_rhat)*(_m2@_rhat)-_m1@_m2)/_rlen**3
        
        print('energy:',self.energy)
        print('')
        
    def calc_force(self):
        
        _m1 = self.m1; _m2 = self.m2; _rhat = self.rhat; _rlen = self.rlen
        self.f1 = c*3/_rlen**4 *( (_m1@_rhat)*_m2 + (_m2@_rhat)*_m1 + (_m1@_m2)*_rhat
                                  - 5*(_m1@_rhat)*(_m2@_rhat)*_rhat ) 
        self.f2 = -self.f1 
        
        print('f1:',self.f1)
        print('f2:',self.f2)
        print('')
        
        
    def calc_energy_expansion(self):
        
        _m1 = self.m1; _m2 = self.m2; _r0hat = self.r0hat; _r0len = self.r0len
        _u1 = self.u1; _u2 = self.u2

        self.U0 = -c*(3*(_m1@_r0hat)*(_m2@_r0hat)-_m1@_m2)/_r0len**3
        
        _f1 = c*3/_r0len**4 *( (_m1@_r0hat)*_m2 + (_m2@_r0hat)*_m1 + (_m1@_m2)*_r0hat
                                  - 5*(_m1@_r0hat)*(_m2@_r0hat)*_r0hat ) 
        _f2 = -_f1
        self.U1 = _u1@_f1 + _u2@_f2
        
        
        self.fc = np.zeros((4,3,3),dtype=float) # 11, 12, 21, 22 by 3x3
        
        _m12 = np.outer(_m1,_m2)+np.outer(_m2,_m1)
        _mr1 = np.outer(_m1,_r0hat)+np.outer(_r0hat,_m1)
        _mr2 = np.outer(_m2,_r0hat)+np.outer(_r0hat,_m2)
        _r12 = np.outer(_r0hat,_r0hat)
        _I = np.eye(3)
        
        _m1m2 = _m1@_m2
        _m1r = _m1@_r0hat
        _m2r = _m2@_r0hat
        
        # print(_mr2)
        
        self.fc[1] = -(5*_m1r*_mr2 + 5*_m2r*_mr1 + 5*(_m1m2-7*_m1r*_m2r)*_r12 
                       -(_m1m2-5*_m1r*_m2r)*_I - _m12)/_r0len**5
        self.fc[2] = -(5*_m1r*_mr2 + 5*_m2r*_mr1 + 5*(_m1m2-7*_m1r*_m2r)*_r12 
                       -(_m1m2-5*_m1r*_m2r)*_I - _m12)/_r0len**5
        _asr = -self.fc.sum(axis=0)
        
        self.fc[0] = _asr
        self.fc[3] = _asr
        _fc = self.fc
        
        self.U2 = _u1@_fc[0,...]@_u1 + _u1@_fc[1,...]@_u2 + _u2@_fc[2,...]@_u1 + _u2@_fc[3,...]@_u2
                
        print('U0:',self.U0)
        print('U1:',self.U1)        
        print('U2:',self.U2)
        print('Uh~',self.U0+self.U1+self.U2)


                
        
    
dip_dip = c_dip_dip(m1=[1,0,0],m2=[1,1,0],u1=[0.01,0.01,0.001],u2=[0,0,0.1])
dip_dip.calc_energy()
dip_dip.calc_force()
dip_dip.calc_energy_expansion()


# nphi = 101
# phi = np.linspace(0,2*np.pi,nphi)
# ntheta = 101
# theta = np.linspace(0,np.pi,ntheta)

# U = np.zeros((ntheta,nphi),dtype=float)
# F = np.zeros((ntheta,nphi,3),dtype=float)

# for ii, _theta in enumerate(theta):
#     for jj, _phi in enumerate(phi):

#         _mx = np.sin(_theta)*np.cos(_phi)
#         _my = np.sin(_theta)*np.sin(_phi)
#         _mz = np.cos(_theta)

#         m2 = np.array([_mx,_my,_mz],dtype=float)
        
#         dip_dip = c_dip_dip(m2=m2)
#         dip_dip.calc_energy()
#         dip_dip.calc_force()

#         U[ii,jj] = dip_dip.energy
        
#         F[ii,jj,:] = dip_dip.f1


# fig, ax = plt.subplots(figsize=(6.5,6))
# extent = [theta.min()/np.pi,theta.max()/np.pi,
#             phi.min()/2/np.pi,phi.max()/2/np.pi]
# vmin = U.min(); vmax = U.max()
# im = ax.imshow(U.T,extent=extent,aspect='auto',origin='lower',interpolation='none',
#                cmap='bwr',vmin=vmin,vmax=vmax)
# fig.colorbar(im,ax=ax,extend='both')
# ax.set_xlabel(r'$\theta/\pi$')
# ax.set_ylabel(r'$\phi/(2\pi)$')
# plt.show()

# fig, ax = plt.subplots(1,3,figsize=(12,6))
# extent = [theta.min()/np.pi,theta.max()/np.pi,
#             phi.min()/2/np.pi,phi.max()/2/np.pi]
# vmin = F.min(); vmax = F.max()
# im = ax[0].imshow(F[...,0].T,extent=extent,aspect='auto',origin='lower',interpolation='none',
#           cmap='bwr')#,vmin=vmin,vmax=vmax)
# fig.colorbar(im,ax=[ax[0]],extend='both',location='top')
# im = ax[1].imshow(F[...,1].T,extent=extent,aspect='auto',origin='lower',interpolation='none',
#           cmap='bwr')#,vmin=vmin,vmax=vmax)
# fig.colorbar(im,ax=[ax[1]],extend='both',location='top')
# im = ax[2].imshow(F[...,2].T,extent=extent,aspect='auto',origin='lower',interpolation='none',
#           cmap='bwr')#,vmin=vmin,vmax=vmax)
# fig.colorbar(im,ax=[ax[2]],extend='both',location='top')
# fig.supxlabel(r'$\theta/\pi$',y=0.04)
# ax[0].set_ylabel(r'$\phi/(2\pi)$')
# plt.show()










