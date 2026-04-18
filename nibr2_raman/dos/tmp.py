import numpy as np
import matplotlib.pyplot as plt

e, dos, _ = np.loadtxt('runo_DS2_DOS_TOTAL',unpack=True)

plt.plot(e,dos)
plt.axvline(0.05146412)
plt.show()