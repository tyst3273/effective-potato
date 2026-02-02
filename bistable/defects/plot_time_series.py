
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

    return n, x, v, y, z

# --------------------------------------------------------------------------------------------------

def get_j_data(filename):
        
    with h5py.File(filename,'r') as db:

        n = db['n'][...]
        x = db['x'][...]

        j = db['j'][...]
        y = db['y'][...]
        z = db['z'][...]

    return n, x, j, y, z

# --------------------------------------------------------------------------------------------------

def get_const_v_data(v,y,z):
    
    filename =  f'results_v_y_{y:.3f}_z_{z:.3f}_low.h5'
    _n, _x, _v, _, _ = get_v_data(filename)

    _ind = np.abs(v-_v).argmin()
    n = _n[_ind,0]
    x = _x[_ind,0]
    j = v * n / x

    return n, x, j

# --------------------------------------------------------------------------------------------------

def get_const_j_data(j,y,z):
    
    filename =  f'results_j_y_{y:.3f}_z_{z:.3f}_low.h5'
    _n, _x, _j, _, _ = get_j_data(filename)

    _ind = np.abs(j-_j).argmin()
    n = _n[_ind,0]
    x = _x[_ind,0]
    v = j * x / n

    return n, x, v

# --------------------------------------------------------------------------------------------------

def get_time_series(y,z):
        

    num_step = 2000

    v_out = -1*np.ones(2*num_step)
    n_out = -1*np.ones(2*num_step)
    x_out = -1*np.ones(2*num_step)
    j_out = -1*np.ones(2*num_step)

    mode = 'const_v'
    v_set = np.linspace(0,0.5,num_step)

    j_thresh = 0.005
    j_set = np.ones(num_step) * j_thresh
    _num = num_step // 2
    j_set[_num:] = np.linspace(j_thresh,0.0,num_step-_num)

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

                print('v:',v)
                print('j:',j)

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

    # with h5py.File('time_series.h5','w') as db:

    #     db.create_dataset('time',data='time')
    #     db.create_dataset('v',data='v_out')
    #     db.create_dataset('n',data='n_out')
    #     db.create_dataset('x',data='x_out')
    #     db.create_dataset('j',data='j_out')

    return time, v_out, n_out, x_out, j_out

# --------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1,3,figsize=(8,3),gridspec_kw={'wspace':0.2})
tax = [ax[0].twinx(),ax[1].twinx(),ax[2].twinx()]

y = 0.1
z = 0.1
time, v, n, x, j = get_time_series(y,z)
tax[0].plot(time,v,c='b',lw=2,ls='-')
ax[0].plot(time,j,c='r',lw=2,ls='-')
ax[0].annotate(f'y={y:.3f}',xy=(0.45,0.9),xycoords='axes fraction',c='k')
ax[0].annotate(f'z={z:.3f}',xy=(0.45,0.825),xycoords='axes fraction',c='k')
ax[0].annotate(f'(a)',xy=(0.05,0.9),xycoords='axes fraction',c='k')

y = 0.1
z = 1.0
time, v, n, x, j = get_time_series(y,z)
tax[1].plot(time,v,c='b',lw=2,ls='-')
ax[1].plot(time,j,c='r',lw=2,ls='-')
ax[1].annotate(f'y={y:.3f}',xy=(0.45,0.9),xycoords='axes fraction',c='k')
ax[1].annotate(f'z={z:.3f}',xy=(0.45,0.825),xycoords='axes fraction',c='k')
ax[1].annotate(f'(b)',xy=(0.05,0.9),xycoords='axes fraction',c='k')

y = 0.25
z = 0.1
time, v, n, x, j = get_time_series(y,z)
tax[2].plot(time,v,c='b',lw=2,ls='-')
ax[2].plot(time,j,c='r',lw=2,ls='-')
ax[2].annotate(f'y={y:.3f}',xy=(0.45,0.9),xycoords='axes fraction',c='k')
ax[2].annotate(f'z={z:.3f}',xy=(0.45,0.825),xycoords='axes fraction',c='k')
ax[2].annotate(f'(c)',xy=(0.05,0.9),xycoords='axes fraction',c='k')

tax[0].set_ylabel('v',color='b')
ax[2].set_ylabel('j',color='r')

# tax[0].axis([0,1,0.0,0.25])
# ax[0].axis([0,1,0,0.01])

for ii in range(3):

    tax[ii].axis([0,1,0.0,0.25])
    ax[ii].axis([0,1,0,0.01])

    ax[ii].set_xlabel('t')

    tax[ii].tick_params(axis='y', colors='b')
    ax[ii].tick_params(axis='y', colors='r')

    ax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')
    tax[ii].axhline(0,lw=1,ls=(0,(1,1)),c='k')

    tax[ii].yaxis.set_label_position("left")
    tax[ii].yaxis.tick_left()

    ax[ii].yaxis.set_label_position("right")
    ax[ii].yaxis.tick_right()

ax[0].set_yticklabels([])
ax[1].set_yticklabels([])
tax[1].set_yticklabels([])
tax[2].set_yticklabels([])

plt.savefig(f'time_series.png',dpi=200,format='png',bbox_inches='tight')
plt.show()
    