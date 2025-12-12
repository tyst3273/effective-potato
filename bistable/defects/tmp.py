

import numpy as np
import matplotlib.pyplot as plt

beta = 4

alpha = 1
gamma = 1.1


x = np.linspace(0,1,1000)
dx = x[1]-x[0]

K = np.exp(-1/x) / beta / x**(beta+1)
J = 1 + alpha*gamma * np.exp((1-gamma)/x) 
H = J * K


# H = np.nan_to_num(H,nan=0)
# K = np.nan_to_num(K,nan=0)
# ind = np.argmax(H)
# print(x[ind])
# ind = np.argmax(K)
# print(x[ind])
# print(gamma/(beta+gamma))


# plt.plot(x,H,c='r',label='H')
# plt.plot(x,J,c='b',label='J')
# plt.plot(x,K,c='g',label='K')

diff = np.diff(H) / dx
deriv = -(beta+1)/beta/x**(beta+2) * ( np.exp(-1/x) + alpha*gamma*np.exp(-gamma/x) ) + \
        1/beta/x**(beta+2) / x * ( np.exp(-1/x) + alpha*gamma**2*np.exp(-gamma/x) )

plt.plot(x[1:],diff,c='r')
plt.plot(x,deriv,c='b')


# plt.axvline(gamma/(beta+1))

plt.legend()
# plt.yscale('log')
plt.axis([0,1,-50,150])

plt.show()