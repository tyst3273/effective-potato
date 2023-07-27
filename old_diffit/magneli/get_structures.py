
from mp_api.client import MPRester 

with MPRester('5ol7p7v2KSJ9z9MZA5D7Xk2FR5uqOYUh') as mpr:   
    ti4o7 = mpr.get_structure_by_material_id('mp-12205')
    ti4o7.to(fmt='poscar', filename='POSCAR_Ti4O7')

    ti5o9 = mpr.get_structure_by_material_id('mp-748')
    ti5o9.to(fmt='poscar', filename='POSCAR_Ti5O9')



