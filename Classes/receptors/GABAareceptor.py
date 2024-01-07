import numpy as np
from .receptor import Receptor

class GABAaReceptor(Receptor):
    def __init__(self, e, g, N_num, loop, alphant, betant):
        super().__init__(e, g, N_num, loop)
        self.alphant = alphant  # Forward binding rate
        self.betant = betant    # Backward binding rate

    def compute_current(self, V_m):
        # Compute the dynamics specific to GABAa receptors
        self.compute_receptor_dynamics(self.alphant, self.betant)

        # Calculate the GABAa current
        self.current_matrix = self.g * self.r * (self.e - V_m)
        return self.current_matrix
