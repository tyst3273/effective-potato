
import gpib_ctypes
gpib_ctypes.gpib.gpib._load_lib(filename)

import pyvisa as pv

rm = pv.ResourceManager()

print(rm.list_resources())