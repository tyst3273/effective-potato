import numpy as np
import matplotlib.pyplot as plt
import h5py

def remove_bose(sig,E,T,axis):
    bose = 1+1/(np.exp(np.abs(E)/(0.08617*T))-1)
    if axis == 0:
        bose = np.tile(bose.reshape(E.size,1),reps=(1,sig.shape[1]))
    else:
        bose = np.tile(bose.reshape(1,E.size),reps=(sig.shape[0],1))
    return sig/bose


def get_data(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines()
    
    e = []
    de = []
    w = []
    dw = []
    q = []

    for ii in range(len(lines)):
        l = lines[ii].strip().split(',')[:-1]
        _q = l[0].strip('.txt').split('_')[2]
        q.append(_q)
        l = l[1:]
        e.append(l[::2])
        de.append(l[1::2])

    q = np.array(q,dtype=float).flatten()
    e = np.array(e,dtype=float).flatten()
    de = np.array(de,dtype=float).flatten()
    
    inds = np.argsort(q)
    q = q[inds]
    e = e[inds]
    de = e[inds]

    return q, e, de


fig, ax = plt.subplots(figsize=(6,6))


f_300K = 'no_couple_6k0_300K.hdf5'

vmin = 0.00001
vmax = 0.00012
bose = True

with h5py.File(f_300K,'r') as db:
    H = db['Dim_0'][...]
    E = db['Dim_1'][...]
    s300 = db['signal'][...]
extent = [H.min(),H.max(),E.min(),E.max()]

with h5py.File(f_300K,'r') as db:
    s300 = db['signal'][...]
if bose:
    s300 = remove_bose(s300,E,T=300,axis=1)
s300 = np.nan_to_num(s300,nan=-1.0)
im = ax.imshow(s300.T,cmap='Greys',aspect='auto',origin='lower',extent=extent,
                  vmin=vmin,vmax=vmax,interpolation='none')

num_q = 11

path_1 = 'tmp_1/subtr_background/'
path_2 = 'tmp_2/subtr_background/'

q1, e1, de1 = get_data(path_1+'Positions.txt')
q1, w1, dw1 = get_data(path_1+'Widths.txt')
q2, e2, de2 = get_data(path_2+'Positions.txt')
q2, w2, dw2 = get_data(path_2+'Widths.txt')

# FWHM; since errorbars are +-, divide by 2
w1 /= 2
w2 /= 2 

ax.errorbar(q1,e1,w1,marker='o',ms=4,mew=1,c='r',lw=1.5,ls='--') #,mfc='none')
ax.errorbar(1-q2,e2,w2,marker='o',ms=4,mew=1,c='m',lw=1.5,ls='--') #,mfc='none')
ax.errorbar(-q1,e1,w1,marker='o',ms=4,mew=1,c='r',lw=1.5,ls='--') #,mfc='none')
ax.errorbar(-1+q2,e2,w2,marker='o',ms=4,mew=1,c='m',lw=1.5,ls='--') #,mfc='none')

ax.set_xlabel('K [rlu]')
ax.set_ylabel('Energy [meV]')
ax.set_title('Q=(6,K,0)')

ax.axis([-1.5,1.5,55,100])

plt.savefig('test.png',dpi=100,bbox_inches='tight')
plt.show()

