
import numpy as np

# --------------------------------------------------------------------------------------------------

def normalize_column_vecs(vecs):
    for ii in range(vecs.shape[1]):
        vecs[:,ii] /= np.sqrt(np.sum(vecs[:,ii]**2))
    return vecs

# --------------------------------------------------------------------------------------------------

# rutile
rutile_lattice_vectors = [[4.593000, 0.000000, 0.000000], # lattice vectors
                          [0.000000, 4.593000, 0.000000],
                          [0.000000, 0.000000, 2.981000]]
rutile_center = [0.000, 0.000, 0.000] # center of octahedra
rutile_octahedra_vectors = [[-0.8972, 0.8972,-1.4905], # orientation vectors of octahedra (see fig)
                            [-0.8972, 0.8972, 1.4905],
                            [ 1.3993, 1.3993, 0.0000]]
rutile_lattice_vectors = np.array(rutile_lattice_vectors).T # column vec
rutile_center = np.array(rutile_center)
rutile_octahedra_vectors = np.array(rutile_octahedra_vectors).T # column vec
rutile_octahedra_vectors = normalize_column_vecs(rutile_octahedra_vectors)
rutile_lattice_vectors_inv = np.linalg.inv(rutile_lattice_vectors)
rutile_octahedra_vectors_c = rutile_lattice_vectors_inv@rutile_octahedra_vectors # crystal coords
rutile_gramian = rutile_octahedra_vectors.T@rutile_octahedra_vectors

# magneli 
magneli_lattice_vectors = [[5.584062, 0.000000, 0.000000], # lattice vectors
                           [2.266011, 6.754238, 0.000000],
                           [2.142339, 2.391429, 7.855713]]
magneli_center = [0.000, 0.000, 0.000] # center of octahedra
magneli_octahedra_vectors = [[-1.71545, 1.04398, 0.16354], # orientation vectors of octahedra
                             [-0.01802,-0.72827, 1.87740],
                             [ 1.07647, 1.52386, 0.63773]]
magneli_lattice_vectors = np.array(magneli_lattice_vectors).T # column vec
magneli_center = np.array(magneli_center)
magneli_octahedra_vectors = np.array(magneli_octahedra_vectors).T # column vec
magneli_octahedra_vectors = normalize_column_vecs(magneli_octahedra_vectors)
magneli_lattice_vectors_inv = np.linalg.inv(magneli_lattice_vectors)
magneli_octahedra_vectors_c = magneli_lattice_vectors_inv@magneli_octahedra_vectors # crystal coords
magneli_gramian = magneli_octahedra_vectors.T@magneli_octahedra_vectors

# --------------------------------------------------------------------------------------------------

print(rutile_gramian)
print(magneli_gramian)

R = np.linalg.solve(magneli_octahedra_vectors,rutile_octahedra_vectors)
print('\n')
print(R)
print(np.linalg.det(R))
print(R.T@R)






