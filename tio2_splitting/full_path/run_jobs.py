import os
#import numpy as np
import shutil
from timeit import default_timer


class _timer:

    def __init__(self,label='timer'):

        """
        start timer
        """

        self.label = label
        self.start_time = default_timer()


    def stop(self):

        """
        stop timer and print elapsed time
        """

        stop_time = default_timer()
        elapsed = (stop_time-self.start_time)/60
        msg = f'{self.label:24} {elapsed:6.3f} [m]\n'
        print(msg)


# --------------------------------------------------------------------------------------------------

def read_template(in_file='template.inp'):

    with open(in_file,'r') as f_in:
        lines = f_in.readlines()
    return lines

# --------------------------------------------------------------------------------------------------

def write_q_dir(q_pt,lines):

    q_dir = f'q_{q_pt}'
    print('\nnow doing:',q_dir)

    if not os.path.exists(q_dir):
        os.mkdir(q_dir)

    with open(os.path.join(q_dir,'run.inp'),'w') as f_out:
        for line in lines:  
            _ = line.strip().split('#')[0]
            if _.endswith('!Qpt'):
                _ = _.strip('!Qpt')
                _ = _+f'{q_pt:g}\n'
                f_out.write(_)
            else:
                f_out.write(line)

    shutil.copy('run_abinit.sh',q_dir)

# --------------------------------------------------------------------------------------------------

def run_abi_job(q_pt,run_cmd,clean=True):

    rm_cmd = '/usr/bin/rm *DEN* *WF*'

    q_dir = f'q_{q_pt}'
    print('\nnow doing:',q_dir)

    if not os.path.exists(q_dir):
        exit(f'directory {q_dir} not found!')

    cwd = os.getcwd()
    q_dir = os.path.join(cwd,q_dir)

    os.chdir(q_dir)

    print(run_cmd)
    timer = _timer(f'q_{q_pt}')
    os.system(run_cmd)
    timer.stop()

    if clean:
        print(rm_cmd)
        os.system(rm_cmd)

    os.chdir(cwd)

def run_gamma(run_cmd):

    q_dir = f'gamma'
    print('\nnow doing:',q_dir)

    if not os.path.exists(q_dir):
        exit(f'directory {q_dir} not found!')

    cwd = os.getcwd()
    q_dir = os.path.join(cwd,q_dir)

    os.chdir(q_dir)

    print(run_cmd)
    timer = _timer(f'gamma')
    os.system(run_cmd)
    timer.stop()

    os.chdir(cwd)


# --------------------------------------------------------------------------------------------------



# ==================================================================================================
# --------------------------------------------------------------------------------------------------
# ==================================================================================================


run_cmd = '/usr/bin/mpirun -np 8 ' \
    '/home/ty/software/abinit-10.4.7/build/src/98_main/abinit run.inp > log 2> err'

task = 'make_directories'
#task = 'run_jobs'

#q_pts = np.arange(2,33)
q_pts = range(2,51)

if task == 'make_directories':
    lines = read_template()
    for q_pt in q_pts:  
        write_q_dir(q_pt,lines)

elif task == 'run_jobs':
    run_gamma(run_cmd)
    for q_pt in q_pts:
        run_abi_job(q_pt,run_cmd)

else:
    exit('fuck!')


