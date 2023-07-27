
import numpy as np
from diffit.m_code_utils import crash, c_timer



class c_exponential_decay_1d:
    
    # ----------------------------------------------------------------------------------------------

    def __init__(self,system_len,num_points,correlation_len,amplitude=1):

        """
        generate an exponentially decaying distribution of variables. 'num_points' sets the number
        of grid points between 0 and 'system_len'. and 'correlation_len' is self explanatory. 
        'amplitude' is the height of the correlation function

        note that if correlation_len is very large, then the tail might spill over the boundaries
        of the system and this can cause errors. 
        """

        self.num_points = np.int(num_points)
        self.system_len = np.float(system_len)
        self.correlation_len = np.float(correlation_len)
        if self.correlation_len <= 0.0:
            msg = 'cannot have negative or 0.0 correlation_ len!\n'
            crash(msg)

        self._generate_ref()

    # ----------------------------------------------------------------------------------------------

    def _generate_ref(self):

        """
        generate array of reference data to create sample distribution
        """
    
        self.dx = self.system_len/self.num_points
        self.pos = np.arange(-self.system_len/2,self.system_len/2,self.dx)

        self.ref_correlation = np.exp(-np.abs(self.pos/self.correlation_len))

        print(self.pos)
        
    # ----------------------------------------------------------------------------------------------

