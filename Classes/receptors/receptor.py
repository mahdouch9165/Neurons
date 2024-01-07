import numpy as np
from ..currents.current import Current

class Receptor(Current):
    def __init__(self, e, g, N_num, loop):
        super().__init__(e, g, N_num, loop)
        self.r = np.zeros((N_num, loop))
        self.nt_release = np.zeros((N_num, loop))

    def update_nt_release(self, new_nt_release):
        self.nt_release = new_nt_release

    def compute_receptor_dynamics(self, alpha, beta):
        for i in range(self.loop - 1):
            drdt = (alpha * self.nt_release[:, i] * (1 - self.r[:, i])) - (beta * self.r[:, i])
            self.r[:, i + 1] = self.r[:, i] + drdt

    def compute_current(self, V_m):
        raise NotImplementedError
