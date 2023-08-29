
import numpy as np
import matplotlib.pyplot as plt

kB = 0.08617333262 # meV/K

def f(E,T):
    return 2*np.sqrt(E/np.pi)*(1/(kB*T))**(3/2)*np.exp(-E/(kB*T))



T = [10,100,300,500,1000]

nE = 1000
E = np.linspace(0,250,1000)

for _T in T:
    plt.plot(E,f(E,_T),label=f'{_T}')

plt.xlabel('E [meV]')
plt.ylabel('f(E)')
plt.legend()
plt.axis([0,150,0,0.08])

plt.show()


