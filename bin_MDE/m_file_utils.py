
"""
Author: Tyler C. Sterling
Email: ty.sterling@colorado.edu
Affil: University of Colorado Boulder, Raman Spectroscopy and Neutron Scattering Lab
Date: 04/26/2023
Description:
    tools to:
        - programatically get data from mantid MDE files (in nexus format) using mantid
            and write to custom hdf5 for quick access from phonon explorer
        - calculate and subtract background from raw data in custom hdf5 file
"""

from timeit import default_timer
import os

# --------------------------------------------------------------------------------------------------
# auxilliary utilities for classes below
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
        elif units == 'ms':
            self.units = 'ms'
            self.scale = 1000
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



