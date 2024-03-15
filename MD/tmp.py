import numpy as np
import matplotlib.pyplot as plt


def calc_fc(r,sigma,epsilon):
    ind = np.flatnonzero(r == 0.0)
    r[ind] = 1.0
    fc = 24*epsilon/r**2*(-26*(sigma/r)**12+7*(sigma/r)**6)
    r[ind] = 0.0
    fc[ind] = 0.0
    fc[ind] = -np.sum(fc) # acoustic sum rule
    return fc

def get_dynmat(q,r,fc):
    dynmat = np.sum(np.exp(2j*np.pi*q*r)*fc)
    return np.real(dynmat)

def get_freq(freq_sq):
    mask = -1*(freq_sq <= 0.0).astype(float)+(freq_sq >= 0).astype(float)
    freq = np.sqrt(np.abs(freq_sq))*mask
    return freq

## get force constants
nr = 25
sigma = 1 #1/2**(1/6)
epsilon = 1

r = np.arange(nr).astype(float)-nr//2 
fc = calc_fc(r,sigma,epsilon)

msg = ''
for ii in range(nr):
    msg += f'{r[ii]: 3.2f} {fc[ii]: 9.6f}\n'
print(msg)

## get phonon freqs
nq = 101
q = np.linspace(-0.5,0.5,nq)
freq_sq = np.zeros(nq)
for ii in range(nq):
    freq_sq[ii] = get_dynmat(q[ii],r,fc)
freq = get_freq(freq_sq)

import matplotlib.pyplot as plt
plt.plot(q,freq)
plt.show()

