
import numpy as np
import matplotlib.pyplot as plt
import h5py


with h5py.File('results_v_sweep.h5','r') as db:

    v = db['v'][...]
    n_lo = db['n_lo'][...]
    n_hi = db['n_hi'][...]
    x_lo = db['x_lo'][...]
    x_hi = db['x_hi'][...]

fig, ax = plt.subplots(figsize=(4.5,4.5))
tax = ax.twinx()

ax.plot(v,n_lo,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1,label='n')
ax.plot(v,n_hi,c='b',lw=1,ls=(0,(2,1)),marker='o',ms=1)

tax.plot(v,x_lo,c='r',lw=1,ls=(0,(2,1)),marker='o',ms=1)
tax.plot(v,x_hi,c='r',lw=1,ls=(0,(2,1)),marker='o',ms=1,label='x')

ax.axhline(0,lw=1,ls=(0,(2,2)),c='k')

ax.set_yscale('log')
tax.set_yscale('log')

ax.set_xlabel('v')
ax.set_ylabel('n',c='b')
tax.set_ylabel('x',c='r')

# tax.spines['right'].set_color('r')
tax.tick_params(axis='y',color='r',labelcolor='r')

# ax.spines['left'].set_color('b')
ax.tick_params(axis='y',color='b',labelcolor='b')

# _n = 0.15
# ax.axis([0,0.5,-_n * 0.05,_n])
# _x = 0.6
# tax.axis([0,0.5,-_x * 0.05,_x])

plt.show()
