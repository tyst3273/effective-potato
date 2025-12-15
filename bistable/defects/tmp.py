

import numpy as np
import matplotlib.pyplot as plt


n = np.linspace(0,10,1000)


f = n / (1+ 0.5 * n)


plt.plot(n,f)


plt.show()