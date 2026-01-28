
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

# --------------------------------------------------------------------------------------------------

def get_v_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        v = db['v'][...]
        y = db['y'][...]
        z = db['z'][...]

    # x[n == 0.0] = np.nan
    # n[n == 0.0] = np.nan

    return n, x, v, y, z

# --------------------------------------------------------------------------------------------------

def get_j_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    # x[n == 0.0] = np.nan
    # n[n == 0.0] = np.nan

    return n, x, j, y, z

# --------------------------------------------------------------------------------------------------

def get_const_v_data(v,y,z):
    
    filename =  f'results_v_y_{y:.3f}_z_{z:.3f}.h5'
    _n, _x, _v, _, _ = get_v_data(filename)

    _ind = np.abs(v-_v).argmin()
    n = _n[_ind,0]
    x = _x[_ind,0]
    j = v * n / x

    return n, x, j

# --------------------------------------------------------------------------------------------------

def get_const_j_data(v,y,z):
    
    filename =  f'results_j_y_{y:.3f}_z_{z:.3f}.h5'
    _n, _x, _j, _, _ = get_j_data(filename)

    _ind = np.abs(j-_j).argmin()
    n = _n[_ind,0]
    x = _x[_ind,0]
    v = j * x / n

    return n, x, v

# --------------------------------------------------------------------------------------------------

z = 0.1
y = 0.1

num_step = 1000

v_out = -1*np.ones(2*num_step)
n_out = -1*np.ones(2*num_step)
x_out = -1*np.ones(2*num_step)
j_out = -1*np.ones(2*num_step)

mode = 'const_v'
v_set = np.linspace(0,0.215,num_step)

j_thresh = 0.015
j_set = np.linspace(j_thresh,0.0,num_step)

v_count = 0
j_count = 0
count = 0

while True:
    
    if mode == 'const_v':

        v = v_set[v_count]
        n, x, j = get_const_v_data(v,y,z)

        v_out[count] = v
        n_out[count] = n
        x_out[count] = x
        j_out[count] = j

        v_count += 1
        count = v_count+j_count

        if j >= j_thresh:
            mode = 'const_j'
            continue

    if mode == 'const_j':

        j = j_set[j_count]
        n, x, v = get_const_j_data(j,y,z)
            
        v_out[count] = v
        n_out[count] = n
        x_out[count] = x
        j_out[count] = j

        j_count += 1
        count = v_count+j_count

        if j_count == num_step:
            break

ind = np.min(np.flatnonzero(v_out < 0))
time = np.arange(ind).astype(float)
time /= time.max()

v_out = v_out[:ind]
n_out = n_out[:ind]
x_out = x_out[:ind]
j_out = j_out[:ind]

with h5py.File('time_series.h5','w') as db:

    db.create_dataset('time',data='time')
    db.create_dataset('v',data='v_out')
    db.create_dataset('n',data='n_out')
    db.create_dataset('x',data='x_out')
    db.create_dataset('j',data='j_out')

fig, ax = plt.subplots(figsize=(4.5,4.5))
tax = ax.twinx()

ax.plot(time,v_out,c='b',lw=1,ls='-')
tax.plot(time,j_out,c='r',lw=1,ls='-')

ax.set_xlabel('time')
ax.set_ylabel('v')
tax.set_ylabel('j')

ax.axis([0,1,0,0.25])
tax.set_ylim(0,0.02)

ax.axhline(0,lw=1,ls=(0,(1,1)),c='k')
tax.axhline(0,lw=1,ls=(0,(1,1)),c='k')

tax.yaxis.set_label_position("right")
tax.yaxis.tick_right()

plt.savefig(f'time_series.png',dpi=200,format='png',bbox_inches='tight')
plt.show()


    