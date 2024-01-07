# Contents of .\Classes\currents\sodium.py
import numpy as np
from .current import Current

class SodiumCurrent(Current):
    def __init__(self, e, g, dimensions, loop, dt, V_initial):
        super().__init__(e, g, dimensions, loop)
        self.dt = dt
        # Initialize gating variable matrices to their steady-state values
        m_ss = self.alpha_m(V_initial) / (self.alpha_m(V_initial) + self.beta_m(V_initial))
        h_ss = self.alpha_h(V_initial) / (self.alpha_h(V_initial) + self.beta_h(V_initial))
        self.m = np.full(dimensions + (loop,), m_ss)
        self.h = np.full(dimensions + (loop,), h_ss)

    def compute_current(self, V_m, time_index):
        # Check time_index bounds
        if time_index < 0 or time_index >= self.loop - 1:
            raise IndexError("time_index out of bounds")

        # Update gating variables for the next time step
        self.m[..., time_index + 1] = self.update_gating(self.m[..., time_index], self.alpha_m, self.beta_m, V_m)
        self.h[..., time_index + 1] = self.update_gating(self.h[..., time_index], self.alpha_h, self.beta_h, V_m)

        # Compute current for the current time step
        current = self.g * (self.m[..., time_index] ** 3) * self.h[..., time_index] * (self.e - V_m)
        self.current_matrix[..., time_index] = current
        return current


    def update_gating(self, gating_variable, alpha_func, beta_func, V_m):
        return gating_variable + self.dt * (alpha_func(V_m) * (1 - gating_variable) - beta_func(V_m) * gating_variable)


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