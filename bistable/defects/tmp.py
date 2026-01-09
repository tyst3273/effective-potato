import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0,1,1000)
vsq = 1.0
x = 1.0
y = 0.1

plt.plot(n,vsq*x*(x*4-y**4))