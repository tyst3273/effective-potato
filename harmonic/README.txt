
it seems that lammps sucks (sortof). if you define a unitcell with bonds in it, 
lammps wont connect the bonds between unitcells if you create a supercell using
the 'replicate nx ny nz' command. you have to manually construct the supercell
(or use some other tools, e.g. VMD ?). 

