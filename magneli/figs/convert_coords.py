
import numpy as np

# rutile
rlv = [[4.593000, 0.000000, 0.000000], # lattice vectors
       [0.000000, 4.593000, 0.000000],
       [0.000000, 0.000000, 2.981000]]
rc = [0.000, 0.000, 0.000] # center of octahedra
rov = [[-0.8972, 0.8972,-1.4905], # orientation vectors of octahedra (see fig)
       [-0.8972, 0.8972, 1.4905],
       [ 1.3993, 1.3993, 0.0000]]
rlv = np.array(rlv).T # column vec
rc = np.array(rc)
rov = np.array(rov).T # column vec
rlv_inv = np.linalg.inv(rlv)

rov_c = rlv_inv@rov # crystal coords
rgram = rov.T@rov

# magneli 
mlv = [[5.584062, 0.000000, 0.000000],
       [2.266011, 6.754238, 0.000000],
       [2.142339, 2.391429, 7.855713]]
mc = [0.000, 0.000, 0.000] # center of octahedra
mov = [[-1.71545, 1.04398, 0.16354], # orientation vectors of octahedra (see fig)
       [-0.01802,-0.72827, 1.87740],
       [ 1.07647, 1.52386, 0.63773]]
mlv = np.array(mlv).T # column vec
mc = np.array(mc)
mov = np.array(mov).T # column vec
mlv_inv = np.linalg.inv(mlv)

mov_c = mlv_inv@mov # crystal coords
mgram = mov.T@mov

rgram /= rgram.max()
mgram /= mgram.max()



print(rgram)
print(mgram)



