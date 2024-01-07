# h_current.py
import numpy as np
from .current import Current

class HCurrent(Current):
    def __init__(self, e, g, N_num, loop, dt):
        super().__init__(e, g, N_num, loop)
        self.dt = dt
        # Initializing S1, S2, F1, F2 based on your MATLAB code
        self.S1 = np.zeros((N_num, loop))
        self.S2 = np.zeros((N_num, loop))
        self.F1 = np.zeros((N_num, loop))
        self.F2 = np.zeros((N_num, loop))
        # Constants
        self.k2 = 4 * 10**-4
        self.C = (2.4 * 10**-4 / 0.8)**2  # Ca_i/Ca_crit to the power n

    def compute_current(self, V_m):
        # Update gating variables
        self.S1[:, 1:] = self.S1[:, :-1] + self.dt * (self.alphaS(V_m[:, :-1]) * (1 - self.S1[:, :-1] - self.S2[:, :-1]) - self.betaS(V_m[:, :-1]) * self.S1[:, :-1] + self.k2 * (self.S2[:, :-1] - self.C * self.S1[:, :-1]))
        self.S2[:, 1:] = self.S2[:, :-1] + self.dt * (-self.k2 * (self.S2[:, :-1] - self.C * self.S1[:, :-1]))
        self.F1[:, 1:] = self.F1[:, :-1] + self.dt * (self.alphaF(V_m[:, :-1]) * (1 - self.F1[:, :-1] - self.F2[:, :-1]) - self.betaF(V_m[:, :-1]) * self.F1[:, :-1] + self.k2 * (self.F2[:, :-1] - self.C * self.F1[:, :-1]))
        self.F2[:, 1:] = self.F2[:, :-1] + self.dt * (-self.k2 * (self.F2[:, :-1] - self.C * self.F1[:, :-1]))
        # Compute current
        self.current_matrix[:, 1:] = self.g * (self.S1[:, 1:] + self.S2[:, 1:]) * (self.F1[:, 1:] + self.F2[:, 1:]) * (self.e - V_m[:, 1:])
        return self.current_matrix

    @staticmethod
    def H_infin(V):
        return 1.0 / (1.0 + np.exp((V + 68.9) / 6.5))

    @staticmethod
    def tauS(V):
        return np.exp((V + 183.6) / 15.24)

    @staticmethod
    def tauF(V):
        return np.exp((V + 158.6) / 11.2) / (1.0 + np.exp((V + 75) / 5.5))

    @staticmethod
    def alphaS(V):
        return HCurrent.H_infin(V) / HCurrent.tauS(V)

    @staticmethod
    def betaS(V):
        return (1.0 - HCurrent.H_infin(V)) / HCurrent.tauS(V)

    @staticmethod
    def alphaF(V):
        return HCurrent.H_infin(V) / HCurrent.tauF(V)

    @staticmethod
    def betaF(V):
        return (1.0 - HCurrent.H_infin(V)) / HCurrent.tauF(V)
