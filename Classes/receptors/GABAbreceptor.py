import numpy as np
from .receptor import Receptor

class GABAbReceptor(Receptor):
    def __init__(self, e, g, N_num, loop, K_one, K_two, K_three, K_four, Kd):
        super().__init__(e, g, N_num, loop)
        self.K_one = K_one       # Opening rate
        self.K_two = K_two       # Closing rate
        self.K_three = K_three   # Desensitization rate
        self.K_four = K_four     # Resensitization rate
        self.Kd = Kd             # Dissociation constant
        self.s = np.zeros((N_num, loop))  # De-activated fraction of receptors

    def compute_current(self, V_m):
        # Update r and s (fractions of activated and de-activated receptors)
        for i in range(self.loop - 1):
            drdt = self.K_one * self.nt_release[:, i] * (1 - self.r[:, i]) - self.K_two * self.r[:, i]
            self.r[:, i + 1] = self.r[:, i] + drdt
            dsdt = self.K_three * self.r[:, i] - self.K_four * self.s[:, i]
            self.s[:, i + 1] = self.s[:, i] + dsdt

        # Calculate the GABAb current
        activated_receptors = (self.s ** 4) / (self.s ** 4 + self.Kd)
        self.current_matrix = self.g * activated_receptors * (self.e - V_m)
        return self.current_matrix
