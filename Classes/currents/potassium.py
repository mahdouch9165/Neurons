# potassium.py
import numpy as np
from .current import Current

class PotassiumCurrent(Current):
    def __init__(self, e, g, N_num, loop, dt):
        super().__init__(e, g, N_num, loop)
        self.dt = dt  # Time step for the simulation
        # Initialize gating variable matrices
        self.n = np.zeros((N_num, loop))

    def compute_current(self, V_m):
        # Update gating variables
        self.n[:, 1:] = self.n[:, :-1] + self.dt * (self.alpha_n(V_m[:, :-1]) * (1 - self.n[:, :-1]) - self.beta_n(V_m[:, :-1]) * self.n[:, :-1])
        # Compute current
        self.current_matrix[:, 1:] = self.g * (self.n[:, 1:] ** 4) * (self.e - V_m[:, 1:])
        return self.current_matrix

    @staticmethod
    def alpha_n(V):
        return (-0.032 * (V - (-55) - 15)) / (np.exp(-(V - (-55) - 15) / 5) - 1)

    @staticmethod
    def beta_n(V):
        return 0.5 * np.exp(-((V - (-55) - 10) / 40))