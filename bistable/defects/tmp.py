
import matplotlib.pyplot as plt
import numpy as np


start = -6
end = 1
num = 1000

x = np.logspace(start,end,num=num)

plt.plot(np.arange(num),x,c='r')
plt.show()