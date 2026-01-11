import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(0,10,1000)
f = 1/(T+1/T)


plt.plot(T,f)
plt.show()