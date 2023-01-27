

import h5py 
import numpy as np
import os
from timeit import default_timer


# --------------------------------------------------------------------------------------------------

def crash(err_msg=None,exception=None):
    """
    stop execution in a safe way
    """
    msg = '\n*** error ***\n'
    if err_msg is not None:
        msg += err_msg+'\n'
    if exception is not None:
        msg += '\nException:\n'+str(exception)+'\n'
    print(msg)
    raise KeyboardInterrupt

# --------------------------------------------------------------------------------------------------

def check_file(file_name):
    """
    check if the specified file exists
    """
    if not os.path.exists(file_name):
        msg = f'the file:\n  \'{file_name}\' \nwas not found!'
        crash(msg)




# --------------------------------------------------------------------------------------------------

class c_timer:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,label,units='s'):
        """
        small tool for timing and printing timing info
        """

        self.label = label
        if units == 'm':
            self.units = 'm'
            self.scale = 1/60
        else:
            self.units = 's'
            self.scale = 1
        self.start_time = default_timer()

    # ----------------------------------------------------------------------------------------------

    def stop(self):
        """
        stop timer and print timing info
        """

        elapsed_time = default_timer()-self.start_time
        elapsed_time *= self.scale
        msg = f'timing:   {self.label} {elapsed_time:9.5f} [{self.units}]'
        print(msg)

# --------------------------------------------------------------------------------------------------




class access_data_in_hdf5:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,file_name,u=None,v=None,w=None):
        """
        class that gets data from hdf5 file
        """
        check_file(file_name)
        self.file_name = file_name

        self._check_proj(u,v,w)
        self._load_Q_and_E()

        # vector from Q (user) to Q points in file. allocate now to avoid redoing for every cut
        self.Q_to_Qp = np.zeros(self.Q_points.shape)
        self.dist_to_Qp = np.zeros(self.Q_points.shape[0])

    # ----------------------------------------------------------------------------------------------

    def _load_Q_and_E(self):
        """
        get the Q_points and DeltaE array from hdf5 file. only want to have to do this once
        """
        _t = c_timer('load_Q_and_E')
        try:
            with h5py.File(self.file_name,'r') as db:
                self.num_Q_in_file = db['H'].size
                self.Q_points = np.zeros((self.num_Q_in_file,3))
                self.Q_points[:,0] = db['H'][...]
                self.Q_points[:,1] = db['K'][...]
                self.Q_points[:,2] = db['L'][...]
                self.E = db['DeltaE'][...]
                self.num_E_in_file = self.E.size
                _H_bins = db['H_bins'][...]
                _K_bins = db['K_bins'][...]
                _L_bins = db['L_bins'][...]
                _E_bins = db['E_bins'][...]
        except Exception as ex:
            msg = f'couldnt read Q and E from hdf5 file \'{self.file_name}\'.\n' \
                  'check the file and the h5py installation and try again.\n\n' \
                  'see the exception below for more hints.\n'
            crash(msg,ex)

        _Q_max = self.Q_points.max(axis=0); _Q_min = self.Q_points.min(axis=0)
        msg = f'\nfound {self.num_Q_in_file} Q-points and {self.num_E_in_file} E-points in file\n\n'
        msg += 'range of reciprocal space included in file:\n'
        msg += f'H range: {_Q_min[0]: .3f} {_Q_max[0]: .3f} rlu\n'
        msg += f'K range: {_Q_min[1]: .3f} {_Q_max[1]: .3f} rlu\n'
        msg += f'L range: {_Q_min[2]: .3f} {_Q_max[2]: .3f} rlu\n'
        msg += f'E range: {self.E.min(): .3f} {self.E.max(): .3f}\n\n'
        msg += 'binning used to make file: (lo, step, hi)\n'
        msg += f'H bins: {_H_bins[0]: .3f}, {_H_bins[1]: .3f}, {_H_bins[2]: .3f}\n'
        msg += f'K bins: {_K_bins[0]: .3f}, {_K_bins[1]: .3f}, {_K_bins[2]: .3f}\n'
        msg += f'L bins: {_L_bins[0]: .3f}, {_L_bins[1]: .3f}, {_L_bins[2]: .3f}\n'
        msg += f'E bins: {_E_bins[0]: .3f}, {_E_bins[1]: .3f}, {_E_bins[2]: .3f}\n'
        print(msg)

        _t.stop()

    # ----------------------------------------------------------------------------------------------

    def _check_proj(self,u,v,w):
        """
        compare projection from user to whats in the file and print a warning if args are 
        different. just to make sure user knows their cuts are gonna be wrong!
        """
        try:
            with h5py.File(self.file_name,'r') as db:
                u_db = db['u'][...]
                v_db = db['v'][...]
                w_db = db['w'][...]
                dim_names = [db[f'Dim_{_}_name'][...].astype(str) for _ in range(4)]
        except Exception as ex:
            msg = f'couldnt read u,v,w vectors from hdf5 file \'{self.file_name}\'.\n' \
                  'check the file and the h5py installation and try again.\n\n' \
                  'see the exception below for more hints.\n'
            crash(msg,ex)
    
        msg = '\ndim. names:\n'
        msg += f'Dim 0: {dim_names[0]}\n'
        msg += f'Dim 1: {dim_names[1]}\n'
        msg += f'Dim 2: {dim_names[2]}\n'
        msg += f'Dim 3: {dim_names[3]}\n'
        print(msg)

        self._check_proj_vector(u,u_db,'u')
        self._check_proj_vector(v,v_db,'v')
        self._check_proj_vector(w,w_db,'w')

    # ----------------------------------------------------------------------------------------------

    def _check_proj_vector(self,user,db,label):
        """
        compare projection vectors and print warning if they dont agree
        """
        msg = f'projection vector \'{label}\' in file: ['+', '.join([f'{_: .3f}' for _ in db])+']'
        if user is None:
            print(msg)
            return

        for ii in range(3):
            if np.round(user[ii],3) != np.round(db[ii],3):
                msg = '*** WARNING ***\n'
                msg += f'user specified projection for the \'{label}\' vector doesnt match what\n' \
                       'is in the file. I will continue, but I hope you know what you are doing!\n'
                msg += 'user: ['+', '.join([f'{_: .3f}' for _ in user])+']\n'
                msg += 'file: ['+', '.join([f'{_: .3f}' for _ in db])+']\n'
                print(msg)
                return

        print(msg)

    # ----------------------------------------------------------------------------------------------

    def get_signal_and_error(self,Q):
        """
        take Q in rlu, find nearest Qpt in file, and return signal and error arrays
        """
        _t = c_timer('get_signal_and_error')

        _n = 10; _eps = 1e-3

        _Qpts = self.Q_points
        _Q2Qp = self.Q_to_Qp
        _d = self.dist_to_Qp

        _Q2Qp[...] = 0.0
        _d[...] = 0.0

        # get distance from Q(user) to all Qpts in file
        _Q2Qp[:,0] = _Qpts[:,0]-Q[0]
        _Q2Qp[:,1] = _Qpts[:,1]-Q[1]
        _Q2Qp[:,2] = _Qpts[:,2]-Q[2]
        _d[...] = np.sqrt(np.sum(_Q2Qp**2,axis=1))

        # find the closest ones
        _inds = np.argsort(_d)[:_n]

        print(_inds)
        print(_Qpts[_inds])
        print(Q)
        print(_d[_inds])

        print('')
        _t.stop()
        print('')

    # ----------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    hdf5_file_name = 'LSNO25_300K_parallel.hdf5'
    access_tools = access_data_in_hdf5(hdf5_file_name)

    access_tools.get_signal_and_error([0.1,0.1,0.1])







