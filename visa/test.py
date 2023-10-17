from m_pyvisa import c_pyvisa

"""
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
print(config['visa']['resource_id'])
print(config['visa']['write_termination'])
"""

pv = c_pyvisa()

pv.print_available_resources(try_to_connect=True)
pv.open_resource('ASRL33::INSTR','ea_ps')

ps = pv.resources['ea_ps']

ps.write('SYST:LOCK OFF')

arr = ps.query('MEAS:Arr?')

print(ps.query('VOLT?'))
print(ps.write('VOLT 0'))
print(ps.query('VOLT?'))















