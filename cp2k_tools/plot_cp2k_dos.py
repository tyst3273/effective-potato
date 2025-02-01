
import numpy as np
import matplotlib.pyplot as plt

fwhm = 0.005
std_dev = fwhm / 2.355

E, up, up_occ, down, down_occ = np.loadtxt('LSNO-dos-1.dos',unpack=True)


dos_up = np.zeros(E.size)
dos_down = np.zeros(E.size)

E -= 0.43412832699242

for ii, _E in enumerate(E):

    dos_up += up[ii] * np.exp(-0.5*(_E-E)**2/std_dev**2) / np.sqrt(2*np.pi) / std_dev
    
    dos_down += down[ii] * np.exp(-0.5*(_E-E)**2/std_dev**2) / np.sqrt(2*np.pi) / std_dev

plt.plot(E, dos_up, label='up')
plt.plot(E, dos_down, label='down')

plt.legend()

plt.show()
