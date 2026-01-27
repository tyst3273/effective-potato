

import numpy as np

v = 0.218**2
y = 0.01
x = np.linspace(y,10.0,1001)

n = np.exp(-1/x)

a = v * n[0] / x[0]
b = y**4 - x[0]**4
c = a+b
d = a+y**4
print(d-x[0]**4)

f = v * n / x + (y**4 - x**4)
print(a+y**4)
print(a,b)
print(a+b)
print((v * n / x)[0])
print(repr(f[0]))
