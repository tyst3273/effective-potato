
import MDE_tools

import importlib
importlib.reload(MDE_tools)
c_MDE_tools = MDE_tools.c_MDE_tools
c_timer = MDE_tools.c_timer

import numpy as np

MDE_file = 'mde/LSNO25_Ei_120meV_5K.nxs'
MDE_tools = c_MDE_tools(MDE_file)

# P4_2/ncm no. 138 : LTT phase of LBCO, LSNO, etc.  https://www.globalsino.com/EM/page1855.html
#SymmetryOperations = 'x,y,z;-x,-y,z;1/2-y,1/2+x,1/2+z;1/2+y,1/2-x,1/2+z;1/2-x,1/2+y,-z;1/2+x,1/2-y,-z;y,x,1/2-z;-y,-x,1/2-z;1/2-x,1/2-y,1/2-z;1/2+x,1/2+y,1/2-z;y,-x,-z;-y,x,-z;x,-y,1/2+z;-x,y,1/2+z;1/2-y,1/2-x,z;1/2+y,1/2+x,z'

# P4mm no. 99 https://www.globalsino.com/EM/page1880.html
SymmetryOperations = 'x,y,z;-x,-y,z;-y,x,z;y,-x,z;x,-y,z;-x,y,z;-y,-x,z;y,x,z'

E_centers = np.arange(50,101)
#E_centers = [62]
dE = 1
dL = 0.5

# LTT lattice vectors
u = [ 1, 0, 0]
v = [ 0, 1, 0]
w = [ 0, 0, 1]

H_bins = [ -8, 0.05, 8 ]
K_bins = [ -8, 0.05, 8 ]
L_centers = np.arange(0,6.5,0.5)

for E in E_centers: 
    for L in L_centers:
    
        out_file = f'LSNO25_Ei_120meV_5K_E_{E:.2f}meV_L_{L:.2f}.hdf5'
        print('\n\n------------------------------------------\n')
        print(out_file)
        print('\n')

        E_bins = [ E-dE, E+dE ]
        L_bins = [ L-dL, L+dL ]
        MDE_tools.bin_MDE(H_bins,K_bins,L_bins,E_bins,u,v,w,SymmetryOperations=SymmetryOperations)
        MDE_tools.save_to_hdf5(out_file)



