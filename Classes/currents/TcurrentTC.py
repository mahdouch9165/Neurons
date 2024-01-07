# t_current.py
import numpy as np
from .current import Current

class TCurrentTC(Current):
    def __init__(self, e, g, N_num, loop, dt):
        super().__init__(e, g, N_num, loop)
        self.dt = dt
        # Initialize gating variable matrices
        self.m_T = np.zeros((N_num, loop))
        self.h_T = np.zeros((N_num, loop))
        self.d_T = np.zeros((N_num, loop))  # Assuming you need d_T based on your MATLAB code

    def compute_current(self, V_m):
        # Update gating variables
        self.m_T[:, 1:] = self.m_T[:, :-1] + self.dt * (-1 * (1 / self.tauTm(V_m[:, :-1])) * (self.m_T[:, :-1] - self.mT(V_m[:, :-1])))
        self.h_T[:, 1:] = self.h_T[:, :-1] + self.dt * (self.alpha1(V_m[:, :-1]) * (1 - self.h_T[:, :-1] - self.d_T[:, :-1] - self.K(V_m[:, :-1]) * self.h_T[:, :-1]))
        self.d_T[:, 1:] = self.d_T[:, :-1] + self.dt * (self.alpha2(V_m[:, :-1]) * (self.K(V_m[:, :-1]) * (1 - self.h_T[:, :-1] - self.d_T[:, :-1]) - self.d_T[:, :-1]))
        # Compute current
        self.current_matrix[:, 1:] = self.g * self.m_T[:, 1:]**3 * self.h_T[:, 1:] * (self.e - V_m[:, 1:])
        return self.current_matrix

    @staticmethod
    def mT(V):
        return 1.0 / (1.0 + np.exp(-(V + 65) / 7.8))

    @staticmethod
    def tauTm(V):
        return 0.15 * TCurrentTC.mT(V) * (1.7 + np.exp(-(V + 30.8) / 13.5))

    @staticmethod
    def alpha1(V):
        return np.exp(-(V + 162.3) / 17.8) / 0.26

    @staticmethod
    def K(V):
        return np.sqrt(0.25 + np.exp((V + 85.5) / 6.3)) - 0.5

    @staticmethod
    def alpha2(V):
        return 1.0 / (TCurrentTC.tau2(V) * (TCurrentTC.K(V) + 1))

    @staticmethod
    def tau2(V):
        return 62.4 / (1 + np.exp((V + 39.4) / 30))
