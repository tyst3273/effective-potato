import numpy as np
import matplotlib.pyplot as plt
import os


class c_dos:
    
    def __init__(self,path='DOSCAR'):
        self.path = path
        self._read_file()
        self._get_meta_data()
        self._check_spin()
        self._check_lm()
        self._get_fermi_energy()
        
    def _get_meta_data(self):
        self.num_atoms = int(self.lines[0].strip().split()[1])
        self.num_e = int(self.lines[5].strip().split()[2])
        
    def _read_file(self):
        with open(self.path,'r') as f:
            self.lines = f.readlines()
            
    def _check_spin(self):
        if len(self.lines[6].strip().split()) == 3:
            self.spin_polarized = False
        else:
            self.spin_polarized = True
    
    def _check_lm(self):
        self.lm_decomposed = bool(self.lines[0].strip().split()[2])
        if self.lm_decomposed:
            self.lm_labels = ['s','p_y','p_z','p_x','d_xy','d_yz','d_z2-r2','d_xz','d_x2-y2',
             'f_y(3x2-y2)','f_xyz','f_yz2','f_z3','f_xz2','f_z(x2-y2)','f_y(3x2-y2)','f_x(x2-3y2)']
    
    def _get_fermi_energy(self):
        self.fermi_energy = float(self.lines[5].strip().split()[3])
        
    def read_dos(self):
        if self.spin_polarized:
            self._read_spin_polarized_dos()
        else:
            self._read_dos()
        
        # if self.lm_decomposed:
        #     if self.spin_polarized:
        #         self._read_lm_spin_polarized_dos()
        #     else:
        #         self._read_lm_dos()
            
    def _read_spin_polarized_dos(self):
        _l = self.lines[6:6+self.num_e]
        _l = [_.strip().split() for _ in _l]
        _data = np.array(_l,dtype=float)
        self.dos_e = _data[:,0]
        self.dos = _data[:,1:3]
        
    def _read_dos(self):
        _l = self.lines[6:6+self.num_e]
        _l = [_.strip().split() for _ in _l]
        _data = np.array(_l,dtype=float)
        self.dos_e = _data[:,0]
        self.dos = _data[:,1]
        
    def _read_lm_dos(self):
        
        _l = self.lines[6+self.num_e+1].strip().split()
        _n = len(_l)-1
        self.lm_dos = np.zeros((self.num_e,self.num_atoms,_n),dtype=float)
        
        _s = 6+self.num_e+1
        for ii in range(self.num_atoms):
            _l = self.lines[_s:_s+self.num_e]
            _l = [_.strip().split() for _ in _l]
            self.lm_dos[:,ii,:] = np.array(_l,dtype=float)[:,1:]
            _s += self.num_e+1

        
# --------------------------------------------------------------------------------------------------

def get_dirs(path=None):
    
    if path is None:
        path = os.getcwd()
        
    _dirs = os.listdir(path)
    
    dirs = []
    for d in _dirs:
        if os.path.isdir(os.path.join(path,d)):
            dirs.append(d)
    
    if len(dirs) == 0:
        return None

    dirs = sorted(dirs)
    return dirs

# --------------------------------------------------------------------------------------------------

