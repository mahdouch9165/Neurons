import numpy as np
from .receptor import Receptor

class NMDAReceptor(Receptor):
    def __init__(self, e, g, N_num, loop, alphaNMDAr, betaNMDAr):
        super().__init__(e, g, N_num, loop)
        self.alphaNMDAr = alphaNMDAr
        self.betaNMDAr = betaNMDAr

    def compute_current(self, V_m):
        self.compute_receptor_dynamics(self.alphaNMDAr, self.betaNMDAr)
        # Calculate the current
        self.current_matrix = self.g * self.r * (self.e - V_m)
        return self.current_matrix
