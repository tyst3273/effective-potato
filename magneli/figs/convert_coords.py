
import numpy as np

# --------------------------------------------------------------------------------------------------

def shift_coords(coords,vec):
    coords[:,0] += vec[0]
    coords[:,1] += vec[1]
    coords[:,2] += vec[2]
    return coords

# --------------------------------------------------------------------------------------------------

def normalize(coords,scale=1.0):
    for ii in range(coords.shape[0]):   
        norm = np.sqrt(np.sum(coords[ii,:]**2))
        if norm < 1e-6:
            continue
        coords[ii,:] *= scale/norm
    return coords

# --------------------------------------------------------------------------------------------------

def write_xyz(fname,coords,types):
    n = coords.shape[0]
    with open(fname,'w') as f:
        f.write(f'{n}\n---')
        for ii in range(n):
            f.write(f'\n{types[ii]:3} {coords[ii,0]: 12.6f} ' \
                    f'{coords[ii,1]: 12.6f} {coords[ii,2]: 12.6f}')

# --------------------------------------------------------------------------------------------------

def rotate_coords(coords,mat):
    for ii in range(coords.shape[0]):
        coords[ii,:] = mat@coords[ii,:]
    return coords

# --------------------------------------------------------------------------------------------------

# magneli
m_types = np.array(['Ti','O','O','O','O','O','O'],dtype=object)
m_oct = [[ 0.00003,  -0.00010,   0.00002], # coordinates of vertices
         [ 0.01800,   0.72823,  -1.87734],
         [ 1.71551,  -1.04399,  -0.16357],
         [-0.01802,  -0.72827,   1.87740],
         [-1.07648,  -1.52388,  -0.63777],
         [ 1.07647,   1.52386,   0.63773],
         [-1.71545,   1.04398,   0.16354]]
m_orr = [[-1.71545, 1.04398, 0.16354], # orientation vectors of octahedra
         [-0.01802,-0.72827, 1.87740],
         [ 1.07647, 1.52386, 0.63773]]

m_oct = np.array(m_oct)
m_oct = shift_coords(m_oct,-m_oct[0,:])
m_oct = normalize(m_oct,scale=2)

m_orr = np.array(m_orr)
m_orr = shift_coords(m_orr,-m_oct[0,:])
m_orr = normalize(m_orr,scale=2)

# rutile
r_types = np.array(['Zr','O','O','O','O','O','O'],dtype=object)
r_orr = [[ -0.89720, 0.89720, -1.49050], # orientation vectors of octahedra
         [ -0.89720, 0.89720,  1.49050],
         [  1.39930, 1.39930,  0.00000]]
r_oct =[[ 0.00000,   0.00000,   0.00000], # coordinates of vertices
        [ 1.39930,   1.39930,   0.00000],
        [ 0.89720,  -0.89720,  -1.49050],
        [-0.89720,   0.89720,  -1.49050],
        [-0.89720,   0.89720,   1.49050],
        [-1.39930,  -1.39930,   0.00000],
        [ 0.89720,  -0.89720,   1.49050]]

r_oct = np.array(r_oct)
r_oct = shift_coords(r_oct,-r_oct[0,:])
r_oct = normalize(r_oct,scale=2)

r_orr = np.array(r_orr)
r_orr = shift_coords(r_orr,-r_oct[0,:])
r_orr = normalize(r_orr,scale=2)

# --------------------------------------------------------------------------------------------------

#print(m_orr.T@m_orr)
#print(r_orr.T@r_orr)

R = np.linalg.solve(m_orr,r_orr).T
m_oct = rotate_coords(m_oct,R)

print(R)

# --------------------------------------------------------------------------------------------------

# offet coords just to plot
r_oct = shift_coords(r_oct,[6,0,0])
octs = np.r_[m_oct,r_oct]
types = np.r_[m_types,r_types]

#r_orr = np.r_[r_orr,np.zeros((1,3))]
#m_orr = np.r_[m_orr,np.zeros((1,3))]
#r_orr = shift_coords(r_orr,[5,0,0])
#types = ['H','P','O','Ti','H','P','O','Zr']
#octs = np.r_[m_orr,r_orr]

write_xyz('oct.xyz',octs,types)

# --------------------------------------------------------------------------------------------------




