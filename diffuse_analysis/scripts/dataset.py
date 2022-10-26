
from scipy.optimize import least_squares as lsqfit
import matplotlib.pyplot as plt
import numpy as np
import h5py


class dataset:

    def __init__(self,file_name):
        self.file_name = file_name
        self.load()

    def load(self):
        with h5py.File(self.file_name) as db:
            self.signal = db['signal'][...]
            self.h = db['h'][...]
            self.k = db['k'][...]
            self.l = db['l'][...]

    def cut_data(self,h,k,l):
        h_inds = self._get_inds(h,self.h)
        k_inds = self._get_inds(k,self.k)
        l_inds = self._get_inds(l,self.l)
        self.cut = self.signal[h_inds,...]
        self.cut = self.cut[:,k_inds,:]
        self.cut = self.cut[...,l_inds]
        self.h_cut = self.h[h_inds]
        self.k_cut = self.k[k_inds]
        self.l_cut = self.l[l_inds]

    def _get_inds(self,Q,Q_arr):
        if Q[0] < -0.5:
            inds = np.flatnonzero(Q_arr > Q[0]+1)
            inds = np.union1d(inds,np.flatnonzero(Q_arr <= Q[1]))
        elif Q[1] > 0.5:
            inds = np.flatnonzero(Q_arr > Q[0])
            inds = np.union1d(inds,np.flatnonzero(Q_arr <= Q[1]-1))
        else:
            inds = np.flatnonzero(Q_arr > Q[0])
            inds = np.intersect1d(inds,np.flatnonzero(Q_arr <= Q[1]))
        return inds

    def fit_symmetric_peaks(self,Q,cut,center,width,height,offset):
        
        def obj_func(params,*args):
            Q = args[0]; cut = args[1]
            center = params[0]
            width = params[1]
            sigma = width/(2*np.sqrt(2*np.log(2)))
            height = params[2]
            offset = params[3]
            calc = np.exp(-0.5*((Q-center)/sigma)**2)*height
            calc += np.exp(-0.5*((Q+center)/sigma)**2)*height
            calc += offset
            return calc-cut
        
        args = [Q,cut]
        params = np.array([center,width,height,offset])
        lb = np.array([-0.5,0.001,0.001,-1e6])
        ub = np.array([0.5,1,1e6,1e6])
        opt = lsqfit(fun=obj_func,x0=params,args=args,bounds=(lb,ub),
            ftol=1e-10,gtol=1e-10,verbose=0,max_nfev=4000,xtol=1e-10)
        center, width, height, offset = opt.x
        return center, width, height, offset 






