
import numpy as np
import matplotlib.pyplot as plt

from dataset import dataset


Ql_o_Qh = 3/4.7


file_name = '293K_quenched_summed.hdf5'
ds = dataset(file_name)
ds.load()

dQ = 0.05
dh = dQ
dk = dQ
dl = dQ*Ql_o_Qh

h = -0.17
k = -0.5
ds.cut_data([h-dh,h+dh],
            [k-dk,k+dk],
            [-0.5,0.5])
cut1 = ds.cut
cut1 = np.mean(cut1,axis=(0,1))

h = -0.17
k = 0.5
ds.cut_data([h-dh,h+dh],
            [k-dk,k+dk],
            [-0.5,0.5])
cut2 = ds.cut
cut2 = np.mean(cut2,axis=(0,1))

h = 0.17
k = -0.5
ds.cut_data([h-dh,h+dh],
            [k-dk,k+dk],
            [-0.5,0.5])
cut3 = ds.cut
cut3 = np.mean(cut3,axis=(0,1))

h = 0.17
k = 0.5
ds.cut_data([h-dh,h+dh],
            [k-dk,k+dk],
            [-0.5,0.5])
cut4 = ds.cut
cut4 = np.mean(cut4,axis=(0,1))

cut = (cut1+cut2+cut3+cut4)/4
Q = ds.l_cut

c, w, h, o = ds.fit_symmetric_peaks(Q,cut,0.35,0.1,0.35,0.25)
print('c,w,h,o',c,w,h,o)

s = w/(2*np.sqrt(2*np.log(2)))
calc = np.exp(-0.5*((Q-c)/s)**2)*h
calc += np.exp(-0.5*((Q+c)/s)**2)*h
calc += o

plt.plot(Q,cut,marker='o',ms=7.5,mfc='k',mec='k',color='k')
plt.plot(Q,calc,ms=0,lw=2,color='r')
plt.show()





