# sodium.py
import numpy as np
from .current import Current

class SodiumCurrent(Current):
    def __init__(self, e, g, N_num, loop, dt):
        super().__init__(e, g, N_num, loop)
        self.dt = dt  # Time step for the simulation
        # Initialize gating variable matrices
        self.m = np.zeros((N_num, loop))
        self.h = np.zeros((N_num, loop))

    def compute_current(self, V_m):
        # Update gating variables
        self.m[:, 1:] = self.m[:, :-1] + self.dt * (self.alpha_m(V_m[:, :-1]) * (1 - self.m[:, :-1]) - self.beta_m(V_m[:, :-1]) * self.m[:, :-1])
        self.h[:, 1:] = self.h[:, :-1] + self.dt * (self.alpha_h(V_m[:, :-1]) * (1 - self.h[:, :-1]) - self.beta_h(V_m[:, :-1]) * self.h[:, :-1])
        # Compute current
        self.current_matrix[:, 1:] = self.g * (self.m[:, 1:] ** 3) * self.h[:, 1:] * (self.e - V_m[:, 1:])
        return self.current_matrix

    @staticmethod
    def alpha_m(V):
        return (-0.32 * (V - (-55) - 13)) / (np.exp(-(V - (-55) - 13) / 4) - 1)

    @staticmethod
    def beta_m(V):
        return (0.28 * (V - (-55) - 40)) / (np.exp((V - (-55) - 40) / 5) - 1)

    @staticmethod
    def alpha_h(V):
        return 0.128 * np.exp(-(V - (-55) - 17) / 18)

    @staticmethod
    def beta_h(V):
        return 4 / (1 + np.exp(-(V - (-55) - 40) / 5))