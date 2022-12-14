

import numpy as np
import os


atoms = [0,1,2]
dirs = ['+x','-x','+y','-y']

for aa in atoms:
    for dd in dirs:

        filename = os.path.join(f'{aa}{dd}','force.dat')
        f_data = np.loadtxt(filename,skiprows=9)

        forces = np.loadtxt(
        print(data)





