
import os
import shutil


"""
sh = 'run_abinit.sh'
run = ''

ls = os.listdir()

for _f in ls:

    if _f.endswith('inp'):

        _dir = _f[:-4]
        if not os.path.exists(_dir):
            os.mkdir(_dir)

        shutil.copy(_f,_dir)
        shutil.copy(sh,_dir)

        run += f'cd {_dir} && sbatch {sh} && cd ../ && '

print(run)
"""

"""
sh = 'run_abinit.sh'
cmd = '/projects/tyst3273/software/abinit-10.4.7/build/src/98_main/mrgddb out_DDB *DDB'
run = ''

ls = os.listdir()

for _f in ls:

    if _f.endswith('inp'):

        _dir = _f[:-4]

        run += f'cd {_dir} && {cmd} && cd ../ && '

print(run)
"""


run = ''
ls = os.listdir()
cmd = '/projects/tyst3273/software/abinit-10.4.7/build/src/98_main/anaddb anaddb.abi'

for _f in ls:

    if _f.endswith('inp'):

        _dir = _f[:-4]
        
        shutil.copy('anaddb.abi',_dir)

        run += f'cd {_dir} && {cmd} && cd ../ && '

print(run)

