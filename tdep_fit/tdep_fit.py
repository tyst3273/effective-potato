import numpy as np
import h5py
import os


class c_data_sets:


    def __init__(self,temps,top_dir=None):

        self.temps = temps
        self.num_temps = len(temps)

        if top_dir is None:
            top_dir = os.getcwd()
        else:
            top_dir = os.path.abspath(top_dir)

        self._get_file_names(top_dir)

        for _f in self.files:
            if not os.path.exists(_f):
                exit(f'\nthe file\n  \'{_f}\'\ndoesnt exist\n')
        
        # set up Qpoint grids
        self.Q_precision = 4
        _n = 1001
        _m = 10
        self.H = np.linspace(-_m,_m,_n)
        self.H = np.round(self.H,self.Q_precision)
        self.K = np.linspace(-_m,_m,_n)
        self.K = np.round(self.K,self.Q_precision)
        self.L = np.linspace(-_m,_m,_n)
        self.L = np.round(self.L,self.Q_precision)


    def _get_file_names(self,top_dir):

        self.files = []
        for T in self.temps:
            _f = f'symm_symm_hMAPB_{T}K.nxs'
            _f = os.path.join(top_dir,_f)
            self.files.append(_f)
            print(_f)


    def _get_data(self,Qpts):
        
        self.nQ = Qpts.shape[0]
        _Qinds = np.zeros((self.nQ,3),dtype=int)

        # loop over Qpoints and get the indices
        for ii in range(self.nQ):
            _Qinds[ii,:] = self._get_ind(Qpts[ii,:])

        # go get the data from the files
        self.sQ = np.zeros((self.num_temps,self.nQ))
        for ii, _f in enumerate(self.files):
            print(f'\nreading {self.nQ} Qpoints from file:\n  \'{_f}\'')
            self.sQ[ii,:] = self._get_data_from_file(_f,_Qinds)


    def _get_data_from_file(self,_f,_Qinds):

        _data = np.zeros(self.nQ)
        try:
            with h5py.File(_f) as _db:
                for ii in range(self.nQ):
                    _data[ii] = _db['entry/data/data'][_Qinds[ii,0],_Qinds[ii,1],_Qinds[ii,2]]
        except Exception as _ex:
            print('couldnt read file \'{_f}\'')
            print('here is the exception:\n',_ex)

        return _data


    def _get_ind(self,Q):
        
        Q = np.round(Q,self.Q_precision)

        _Hind = np.flatnonzero(self.H == Q[0])[0]
        _Kind = np.flatnonzero(self.K == Q[1])[0]
        _Lind = np.flatnonzero(self.L == Q[2])[0]
        
        return [_Hind,_Kind,_Lind]








if __name__ == '__main__':


    temps = [200,300]
    dsets = c_data_sets(temps,top_dir='/media/ty/hitachi/MAPB/tdep')

    Qpts = np.array([[-10,-10,-10],
                     [  0,  0,  0],
                     [  1,  2,  3]],dtype=float)
#    dsets._get_data(Qpts)



"""
to-do: need to write methods to come up with sets of Q-points to calculate
rods on

also need methods to get 'background' around each rod
"""








