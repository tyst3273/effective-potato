import numpy as np


class c_md:

    def __init__(self,num_atoms=50,box_size=None,epsilon=1,sigma=1):

        self.num_atoms = num_atoms
        self.epsilon = epsilon
        self.sigma = sigma

        if box_size is None:
            box_size = 2**(1/6)*self.sigma*self.num_atoms

        self.box_size = box_size

        self._setup_box()

    def _setup_box(self):

        self.pos_c = np.linspace(self.num_atoms)
        self.pos_r = self.pos_c/box_size

    def _get_neighbors(self):

        self.delta_pos = 

    def _do_minimum_image(self):

