
import numpy as np

a = np.loadtxt('fc')
b = np.loadtxt('/home/ty/research/repos/elph/tests/octahedra_complex/fc_elph')
np.savetxt('diff',a-b,fmt='% 4.2f')

print(a-b)
