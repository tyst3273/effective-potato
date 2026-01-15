
import matplotlib.pyplot as plt
import numpy as np

nn = 1001
n = np.linspace(0,1,nn)

x = 0.01
j = 0.1
z = 0.01

f = np.exp(-1/x) * np.exp(j*z/n) - n

plt.plot(n,f)
plt.show()