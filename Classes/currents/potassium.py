# Contents of .\Classes\currents\potassium.py
import numpy as np
from .current import Current

class PotassiumCurrent(Current):
    def __init__(self, e, g, dimensions, loop, dt, V_initial):
        super().__init__(e, g, dimensions, loop)
        self.dt = dt
        # Initialize gating variable matrix to its steady-state value
        n_ss = self.alpha_n(V_initial) / (self.alpha_n(V_initial) + self.beta_n(V_initial))
        self.n = np.full(dimensions + (loop,), n_ss)

    def compute_current(self, V_m, time_index):
        # Check time_index bounds
        if time_index < 0 or time_index >= self.loop - 1:
            raise IndexError("time_index out of bounds")

        # Update gating variable for the next time step
        self.n[..., time_index + 1] = self.update_gating(self.n[..., time_index], self.alpha_n, self.beta_n, V_m)

        # Compute current for the current time step
        current = self.g * (self.n[..., time_index] ** 4) * (self.e - V_m)
        self.current_matrix[..., time_index] = current
        return current


    def update_gating(self, gating_variable, alpha_func, beta_func, V_m):
        return gating_variable + self.dt * (alpha_func(V_m) * (1 - gating_variable) - beta_func(V_m) * gating_variable)

    @staticmethod
    def alpha_n(V):
        return (-0.032 * (V - (-55) - 15)) / (np.exp(-(V - (-55) - 15) / 5) - 1)

    @staticmethod
    def beta_n(V):
        return 0.5 * np.exp(-((V - (-55) - 10) / 40))