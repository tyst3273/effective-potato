
import numpy as np
import h5py

from diffit.m_crystal import c_crystal
from diffit.m_code_utils import c_timer, crash
from diffit.m_structure_io import write_xyz, write_poscar, write_lammps_data_file
from diffit.m_domains import c_domains

from diffit.m_PSF_interface import run_PSF



_t = c_timer('run_diffit',units='m')


def source_supercell():
    
    # --------------------------------------------------------------------------------------------------
    
    rng = np.random.default_rng()
    normal = rng.normal(loc=10,scale=1,size=50).round().astype(int)
    
    rutile = c_crystal(poscar='TiO2/POSCAR_TiO2_ideal')
    a = rutile.basis_vectors[0,0]; c = rutile.basis_vectors[2,2]
    
    nx = 40; nz = 60
    rutile.build_supercell([nx,nx,nz])
    
    # distances between O planes 
    interplanar_distance = 1.0402232469225114 
    
    # shear plane orientation vectors
    orientation_vectors = np.loadtxt('vectors/CS_132_vectors')
    orientation_vectors[:,0] /= a
    orientation_vectors[:,1] /= a
    orientation_vectors[:,2] /= c
    num_vectors = orientation_vectors.shape[0]
    for ii in range(num_vectors):
        orientation_vectors[ii,:] /= np.sqrt(orientation_vectors[ii,:]@orientation_vectors[ii,:])
    
    # shear planes pass thru this coordinate
    origins = np.loadtxt('vectors/CS_132_origins')
    origins[:,0] *= a
    origins[:,1] *= a
    origins[:,2] *= c
    
    # possible displacement vectors for the shear
    displacement_vectors = np.loadtxt('vectors/CS_132_displacements')
    displacement_vectors[:,0] *= a/2
    displacement_vectors[:,1] *= a/2
    displacement_vectors[:,2] *= c/2
    
    # --------------------------------------------------------------------------------------------------
    
    defects = []
    
    domains = c_domains(rutile)
    
    epsilon = 0.005
    
    num_cluster = np.arange(1,3)
    vector_inds = np.arange(num_vectors)
    
    x_inds = np.arange(nx)
    y_inds = np.arange(nx)
    z_inds = np.arange(nz)
    
    # max_defects = -1
    num_defects = 10
    
    # delete planes of O atoms 
    for ii in range(num_defects):
    
#        np.random.shuffle(vector_inds)
        ind = vector_inds[0]
    
        vector = orientation_vectors[ind,:]
        delta = vector*epsilon
    
        np.random.shuffle(x_inds)
        np.random.shuffle(y_inds)
        np.random.shuffle(z_inds)
        origin = origins[ind,:]
        origin += [x_inds[0]*a,y_inds[0]*a,z_inds[0]*c]
        # origin += [nx//2*a,nx//2*a,nz//2*c]
    
        displacement = displacement_vectors[ind,:]
    
        np.random.shuffle(num_cluster)
        num = num_cluster[0]
    
        for ii in range(-num,num):

            np.random.shuffle(normal)
            n = normal[0]
    
            offset = n*ii*vector*interplanar_distance
            
            # get the O atoms
            delete = domains.find_slab(origin+offset-delta,vector,2*epsilon)
            if delete.size == 0:
                print('\nempty!\n')
                continue
    
            # check that we didnt find any Ti atoms
            nums = domains.crystal.sc_type_nums[delete]
            if np.any(nums == 0):
                msg = 'found Ti atoms in the plane to delete...'
                crash(msg)
    
            # delete the atoms
            print(f'\ndeleting {delete.size} O atoms!\n')
            domains.crystal.delete_atoms(delete)
            #domains.replace_slab_types('C')
    
            defects.append({'vector':vector,'origin':origin+offset,'displacement':displacement})
    
    max_defects = len(defects)
    
    # get the sets of inds to displace (have to do after deleting O atoms ...)
    for ii, defect in enumerate(defects):
    
        print(f'getting inds for defect {ii} out of {max_defects}')
        
        vector = defect['vector']
        origin = defect['origin']
    
        inds = domains.find_slab(origin,vector)
        defect['inds'] = inds
    
    # now go and displace the slabs
    for ii, defect in enumerate(defects):
    
        print(f'displacing defect {ii} out of {max_defects}')
     
        inds = defect['inds']
        displacement = defect['displacement']
    
        domains.merge_slab_inds(inds)
        domains.displace_slab(displacement,update_reduced=False)
    
    rutile = domains.get_crystal()
    rutile.update_reduced_coords()

    write_xyz('full_supercell.xyz',rutile)
    
    # --------------------------------------------------------------------------------------------------
    # crop a rutile shaped chunk of the crystal
    
    nxc = 20; nzc = 30
    origin = [10*a,10*a,10*c]
    vectors = [[nxc*a,    0,    0],
               [    0,nxc*a,    0],
               [    0,    0,nzc*c]]
    
    domains.crop_crystal(origin,vectors,epsilon=0.0)
    
    rutile = domains.get_crystal()

    write_xyz('cropped_supercell.xyz',rutile)
    
    # # need to 'pad' lattice vectors so that atoms near boundary arent overlapping
    # rutile.sc_vectors[0,0] += a/4
    # rutile.sc_vectors[1,1] += a/4
    # rutile.sc_vectors[2,2] += c/2
    
    # write_xyz('cropped_supercell.xyz',rutile)

    # num_Ti = np.count_nonzero(rutile.sc_type_nums == 0)
    # num_O = np.count_nonzero(rutile.sc_type_nums == 1)

    # Z_Ti = 2.196
    # Z = Z_Ti*num_Ti
    # Z_O = -Z/num_O

    # write_lammps_data_file('lammps.MA.in',rutile,atom_masses=[47.867,15.999],
    #         atom_charges=[Z_Ti,Z_O])
    
    return rutile

# --------------------------------------------------------------------------------------------------        

nx = 20; nz = 30
psf_kwargs = {'atom_types':['Ti','O'],
              'experiment_type':'neutrons',
              'num_Qpoint_procs':16,
              'Qpoints_option':'mesh',
              'Q_mesh_H':[ 2.5, 3.5, nx+1],
              'Q_mesh_K':[ 0.5, 1.5, nx+1],
              'Q_mesh_L':[-0.5, 0.5, nz+1],
              'output_prefix':None}
sq = np.zeros((nx+1,nx+1,nz+1),dtype=float)

num_avg = 1
for ii in range(num_avg):

    supercell = source_supercell()
    _sq, H, K, L = run_PSF(supercell,**psf_kwargs)
    sq += _sq
    
sq /= num_avg

with h5py.File('magneli_STRUFACS.hdf5','w') as db:
    db.create_dataset('sq_elastic',data=sq)
    db.create_dataset('H',data=H)
    db.create_dataset('K',data=K)
    db.create_dataset('L',data=L)
    
_t.stop()




