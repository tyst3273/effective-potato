import numpy as np
import matplotlib.pyplot as plt

beta = 6
gamma = 1

ny = 5000
y = np.linspace(0.01,0.25,ny)
z0 = 0.2

solutions = [[]  for _ in range(ny)]

for ii, yy in enumerate(y):

    print(ii)

    x = np.linspace(yy,1,5000)
    F = x ** beta - yy ** beta - z0 ** 2 * np.exp( - 1 / x ** gamma )

    zeros = np.flatnonzero(np.diff(np.sign(F)))
    solutions[ii] = x[zeros]

fig, ax = plt.subplots(figsize=(4.5,4.5))

for ii, yy in enumerate(y):

    for xx in solutions[ii]:
        ax.scatter(yy,xx,marker='o',s=10,c='k')

plt.show()


