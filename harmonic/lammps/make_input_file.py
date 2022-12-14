
import numpy as np


def write_file(f,natoms,nbonds,nx,ny,a,types,pos,inds):

    with open(f,'w') as fout:
        fout.write('# harmonic 2d perovskite\n\n')
        fout.write(f'{natoms:g} atoms\n{nbonds} bonds\n\n')
        fout.write('2 atom types\n1 bond types\n\n')
        fout.write(f'0 {nx*a:5.3f} xlo xhi\n')
        fout.write(f'0 {ny*a:5.3f} ylo yhi\n')
        fout.write(f'-1 1 zlo zhi\n\n')
        fout.write('Atoms\n\n')
        for ii in range(natoms):
            fout.write(f'{ii+1:3g} 0  {types[ii]:3g}  {pos[ii,0]:5.3f} {pos[ii,1]:5.3f} 0.0\n')
        fout.write('\nBonds\n\n')
        shift = 1
        for ii in range(n1):
            _i = inds[ii]+1
            for jj in range(4):
                _j = nn[ii,jj]+1
                fout.write(f'{shift+jj:3g}  1 {_i:3g}  {_j:3g}\n')
            shift += 4


def get_nn(ii,pos):

    pii = pos[ii,:]
    pii = np.tile(pii.reshape(1,2),reps=(pos.shape[0],1))
    rel = pos-pii

    dr = -(rel[:,0] > 1/2).astype(int)
    dr += (rel[:,0] <= -1/2).astype(int)
    rel[:,0] += dr

    dr = -(rel[:,1] > 1/2).astype(int)
    dr += (rel[:,1] <= -1/2).astype(int)
    rel[:,1] += dr

    dist = np.sqrt(np.sum(rel**2,axis=1))
    inds = np.argsort(dist)
    dist = dist[inds]

    return inds[1:5]


# lattice constant
a = 1.0

# number of reps in each direction
nx = 5
ny = 5

ncells = nx*ny

# unitcell info
types = np.array([1,2,2])
basis = np.array([[0.0,0.0],
                  [0.5,0.0],
                  [0.0,0.5]])
nbasis = basis.shape[0]
natoms = ncells*nbasis

# create supercell and go to reduced coords 
types = np.tile(types,reps=(ncells))
pos = np.tile(basis.reshape(1,nbasis,2),reps=(ncells,1,1))
pos = pos.reshape(ncells*3,2)
gx, gy = np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
gx = gx.flatten(); gy = gy.flatten()
grid = np.array((gx,gy)).T
grid = np.tile(grid.reshape(ncells,1,2),reps=(1,3,1))
grid = grid.reshape(ncells*3,2)
pos = pos+grid
pos[:,0] /= nx; pos[:,1] /= ny

# get nn for each atom in the supercell
inds = np.flatnonzero(types == 1)
n1 = inds.size
nn = np.zeros((n1,4),dtype=int)
for ii in range(n1):
    ind = inds[ii]
    nn[ii,:] = get_nn(ind,pos)
nbonds = nn.size

pos[:,0] *= nx*a
pos[:,1] *= ny*a


write_file('unitcell.prim',natoms,nbonds,nx,ny,a,types,pos,inds)


# now do displacements
d = 0.1
for ii in range(3): # atoms 0, 1, and 2 are the basis

    tmp = np.copy(pos)

    tmp[ii,0] += d
    f = f'{ii}+x.structure'
    write_file(f,natoms,nbonds,nx,ny,a,types,tmp,inds)
    tmp[ii,0] += -2*d
    f = f'{ii}-x.structure'
    write_file(f,natoms,nbonds,nx,ny,a,types,tmp,inds)
    tmp[ii,0] += d

    tmp[ii,1] += d
    f = f'{ii}+y.structure'
    write_file(f,natoms,nbonds,nx,ny,a,types,tmp,inds)
    tmp[ii,1] += -2*d
    f = f'{ii}-y.structure'
    write_file(f,natoms,nbonds,nx,ny,a,types,tmp,inds)
    tmp[ii,1] += d









