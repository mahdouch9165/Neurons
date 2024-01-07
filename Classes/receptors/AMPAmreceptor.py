import numpy as np
from .receptor import Receptor

class AMPAmReceptor(Receptor):
    def __init__(self, e, g, N_num, loop, R_o, R_c, R_d, R_s, Rb):
        super().__init__(e, g, N_num, loop)
        self.R_o = R_o   # Opening rate
        self.R_c = R_c   # Closing rate
        self.R_d = R_d   # Desensitization rate
        self.R_s = R_s   # Resensitization rate
        self.Rb = Rb     # Binding rate
        self.s = np.zeros((N_num, loop))  # De-activated fraction of receptors

    def compute_current(self, V_m):
        # Compute the dynamics specific to AMPA-metabotropic receptors
        for i in range(self.loop - 1):
            drdt = self.R_o * self.nt_release[:, i] * (1 - self.r[:, i]) - self.R_c * self.r[:, i]
            self.r[:, i + 1] = self.r[:, i] + drdt
            dsdt = self.R_d * self.r[:, i] - self.R_s * self.s[:, i]
            self.s[:, i + 1] = self.s[:, i] + dsdt

        # Calculate the AMPA-metabotropic current
        activated_receptors = (self.s ** 2) / (self.s ** 2 + self.Rb)
        self.current_matrix = self.g * activated_receptors * (self.e - V_m)
        return self.current_matrix
