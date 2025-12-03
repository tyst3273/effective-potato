

msg = ''
for num in range(2,51):
    msg += f'cd q_{num} && sbatch run_abinit.sh && cd ../ &&  '
print(msg)
