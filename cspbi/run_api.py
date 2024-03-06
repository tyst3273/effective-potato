
import numpy as np


class c_trajectory:

    def __init__(self,temperature=540,files=[1,2,3],num_steps=501,num_atoms=69120,
            num_reps=24):
        
        self.temperature = temperature
        self.files = files
        self.num_files = len(files)
        self.num_steps = num_steps
        self.num_atoms = num_atoms
        self.

def get_traj_from_npz(temperature,files=[1,2,3]):

    with np.load(fname) as npz:
        

num_trajectory_blocks = 5 # split trajectory into 5 blocks to avg. over
trajectory_blocks = [0,2,4] # only use these 3 of the 5 blocks (see 'files' below)

# options to read from Walsh files
num_steps = 501 # steps in each file
num_atoms = 69120 # number of atoms in the files
num_reps = 24 # number of reps of unitcell to make supercell

# which temps to do
temps = [400,440,460,480,520,540,560,600]

# it's ALOT of data so I dont want to read all files ...
# rather, is use the last 2 split into 5 'blocks' and pick 3 non-consecutive blocks 
# each block is 40 ps, so hopefully is long enough to adequately sample the interesting
# fluctuations 
files = [2,3]
num_files = len(files)


# -------- loop over files and run diffuse calculation for each -----------
for T in temps:

    _dir = f'constant_temp_{T}K'

    box = np.zeros((3,3)) 
    pos = np.zeros((num_steps*num_files,num_atoms,3))
    types = np.zeros(num_atoms,dtype=int)

    # read the files
    for ii in range(num_files):

        fname = _dir+f'/nptraj{ii+1}.npz'
        with np.load(fname) as npz:

            _box = npz['cells'] # steps, 3, 3
            _pos = npz['positions'] # steps, atoms, 3
            _nums = npz['numbers'] # steps, atoms

            box += _box.mean(axis=0)
            types = _nums[0,:]
            pos[ii*num_steps:(ii+1)*num_steps,...] = _pos
    
    # my code expects types to be consecutive integers. see input.py
    types[np.flatnonzero(types == 55)] = 0 # Cs
    types[np.flatnonzero(types == 82)] = 1 # Pb
    types[np.flatnonzero(types == 53)] = 2 # I

    box /= num_files # box vectors (since we arent unwrapping, they aren't used)
    cell = box/num_reps # lattice vectors

    print(box.round(2))
    print(cell.round(2))
    print('\n')





