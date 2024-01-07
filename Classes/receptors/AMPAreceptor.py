import numpy as np
from .receptor import Receptor

class AMPAReceptor(Receptor):
    def __init__(self, e, g, N_num, loop, alphar, betar):
        super().__init__(e, g, N_num, loop)
        self.alphar = alphar  # Forward binding rate
        self.betar = betar  # Backward binding rate

    def compute_current(self, V_m):
        # Compute the dynamics specific to AMPA receptors
        self.compute_receptor_dynamics(self.alphar, self.betar)

        # Calculate the AMPA current
        self.current_matrix = self.g * self.r * (self.e - V_m)
        return self.current_matrix
