
import sys
import os

# for DZVP basis, NGRIDS = 5, CUTOFF = 2000, REL_CUTOFF = 50 are good choices to ~3 meV.
# for SZV basis, NGRIDS = 5, CUTOFF = 2000, REL_CUTOFF = 50 are good choices to ~3 meV.

work_dir = 'work_dir'
cutoffs = [1500, 1750, 2000, 2250, 2500, 2750, 3000] 
rel_cutoffs = [50] 

#run_cmd = f'/home/ty/anaconda3/envs/cp2k/bin/cp2k.psmp -i tmp.inp -o tmp.out'
run_cmd = f'mpirun -np 16 cp2k.psmp -i tmp.inp > tmp.out 2> err'

# to track convergence
last_energy = 0.0

# --------------------------------------------------------------------------------------------------

# read the template file
with open('template.inp','r') as fin:
    template = fin.readlines()

# --------------------------------------------------------------------------------------------------

cwd = os.getcwd()

for cut in cutoffs:

    for rel in rel_cutoffs:

        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        if os.path.exists(os.path.join(work_dir,'tmp.inp')):
            os.remove(os.path.join(work_dir,'tmp.inp'))
        if os.path.exists(os.path.join(work_dir,'tmp.out')):
            os.remove(os.path.join(work_dir,'tmp.out'))

        with open(os.path.join(work_dir,'tmp.inp'),'w') as inp:
            inp.write(f'# CUT {cut}, REL {rel}\n\n')
            for line in template:
                if line.strip().startswith('CUTOFF'):
                    inp.write(line.replace('!cut',f'{cut}'))
                    continue
                if line.strip().startswith('REL_CUTOFF'):
                    inp.write(line.replace('!rel',f'{rel}'))
                    continue
                else:
                    inp.write(line)

        os.chdir(work_dir)

        val = os.system(run_cmd)
        if val != 0:
            print('fuck!')
            exit()

        os.chdir(cwd)

        grid_counts = []

        # parse output file
        with open(os.path.join(work_dir,'tmp.out'),'r') as out:
            out_data = out.readlines()

        for line in out_data:

            # get total energy
            if line.strip().startswith('Total energy:'):
                energy = float(line.split()[-1])*13.605703976
                continue

            elif line.strip().startswith('SUM OF ATOMIC FORCES'):
                forces = line.strip().split()[4:]

            # get grid count
            elif line.strip().startswith('count for grid'):
                grid_counts.append(line.strip().split()[4])
            else:
                continue

        conv = energy-last_energy
        last_energy = energy
        
        # print results
        msg =  f'\ncutoff:      [Ry]  {cut:<}'
        msg += f'\nrel_cutoff:  [Ry]  {rel:<}'
        msg += f'\nenergy:      [eV]  {energy:<}'
        msg += f'\nconvergence: [eV]  {conv:<}'
        msg += f'\nforces:      [au]  ' + ' '.join(f'{f}' for f in forces)
        msg += f'\ngrid_counts:       ' + '  '.join(f'{int(n)}' for n in grid_counts)
        print(msg)
        
        # print(cut,rel,tot_e,'\t',*grid_counts)#,ng5)

# --------------------------------------------------------------------------------------------------


