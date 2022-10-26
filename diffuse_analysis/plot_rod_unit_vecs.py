
import numpy as np
import math

h0 = np.array([0.5,0.1765,0.3345])
h0 = h0/np.sqrt(np.sum(h0**2))

h1 = np.array([0.7481,0.2697,0.5])
h1 = h1/np.sqrt(np.sum(h1**2))

h = h0+h1
h = h/2
print(h)


k0 = np.array([0.1666,0.5,0.33])
k0 = k0/np.sqrt(np.sum(k0**2))

k1 = np.array([0.7481,0.7303,0.5])
k1 = k1/np.sqrt(np.sum(k1**2))

k = k0+k1
k = k/2
print(k)





