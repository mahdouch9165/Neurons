# calcium_t_current_trn.py
import numpy as np
from .current import Current

class TCurrentTRN(Current):
    def __init__(self, e, g, N_num, loop, dt):
        super().__init__(e, g, N_num, loop)
        self.dt = dt
        self.m = np.zeros((N_num, loop))
        self.h = np.zeros((N_num, loop))

    def compute_current(self, V_m):
        self.m[:, 1:] = self.m[:, :-1] + self.dt * (-1.0 / self.time_m_T_RE(V_m[:, :-1]) * (self.m[:, :-1] - self.mINFRE(V_m[:, :-1])))
        self.h[:, 1:] = self.h[:, :-1] + self.dt * (-1.0 / self.time_h_T_RE(V_m[:, :-1]) * (self.h[:, :-1] - self.hINFRE(V_m[:, :-1])))
        self.current_matrix[:, 1:] = self.g * (self.m[:, 1:] ** 2) * self.h[:, 1:] * (self.e - V_m[:, 1:])
        return self.current_matrix

    @staticmethod
    def mINFRE(V):
        return 1.0 / (1.0 + np.exp((-(V + 52) / 7.4)))

    @staticmethod
    def time_m_T_RE(V):
        return 0.44 + (0.15 / (np.exp((V + 27) / 10) + np.exp(-(V + 102) / 15)))

    @staticmethod
    def hINFRE(V):
        return 1.0 / (1.0 + np.exp((V + 80) / 5))

    @staticmethod
    def time_h_T_RE(V):
        return 22.7 + (0.27 / (np.exp((V + 48) / 4) + np.exp(-(V + 407) / 50)))
