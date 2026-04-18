
import os
import shutil


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

