
from m_pyvisa import c_pyvisa


pv = c_pyvisa()

pv.print_available_resources(try_to_connect=False)

pv.open_resource()



ps = pv.resources['ea_ps']

stat = ps.write('VOLT 1.0')

stat = ps.write('VOLT?')
print(ps.read())

stat = ps.write('SYST:LOCK OFF')
print(stat)

"""
ps.write('SYST:LOCK OFF')

arr = ps.query('MEAS:Arr?')

print(ps.query('VOLT?'))
print(ps.write('VOLT 0'))
print(ps.query('VOLT?'))
"""














