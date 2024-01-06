import numpy as np

class Current:
    def __init__(self, g_max, E_rev):
        self.g_max = g_max  # Maximum conductance
        self.E_rev = E_rev  # Reversal potential

    def compute_current(self, V, m, h, n):
        raise NotImplementedError("This method should be implemented by subclasses.")

class SodiumCurrent(Current):
    def compute_current(self, V, m, h, n):
        return self.g_max * m**3 * h * (V - self.E_rev)

class PotassiumCurrent(Current):
    def compute_current(self, V, m, h, n):
        return self.g_max * n**4 * (V - self.E_rev)

class LeakyCurrent(Current):
    def compute_current(self, V, m, h, n):
        return self.g_max * (V - self.E_rev)

class TRNTCurrent(Current):
    def compute_current(self, V, m_T, h_T, n):
        # TRN-specific T-current calculation
        m_inf = 1 / (1 + np.exp(-(V + 52) / 7.4))
        h_inf = 1 / (1 + np.exp((V + 80) / 5))
        tau_m_T = 0.44 + (0.15 / (np.exp((V + 27) / 10) + np.exp(-(V + 102) / 15)))
        tau_h_T = 22.7 + (0.27 / (np.exp((V + 48) / 4) + np.exp(-((V + 407) / 50))))
        return self.g_max * (m_T ** 2) * h_T * (V - self.E_rev)

class TCTCurrent(Current):
    def compute_current(self, V, m_T, h_T, n):
        # TC-specific T-current calculation
        m_inf = 1 / (1 + np.exp(-(V + 65) / 7.8))
        tau_m_T = 0.15 * m_inf * (1.7 + np.exp(-(V + 30.8) / 13.5))
        alpha_1 = np.exp(-(V + 162.3) / 17.8) / 0.26
        K_V = np.sqrt(0.25 + np.exp((V + 85.5) / 6.3)) - 0.5
        tau_2 = 62.4 / (1 + np.exp((V + 39.4) / 30))
        alpha_2 = 1 / (tau_2 * (K_V + 1))
        return self.g_max * m_T ** 3 * h_T * (V - self.E_rev)

class IHCurrent(Current):
    def compute_current(self, V, S1, S2, F1, F2):
        # H-Current calculation
        h_inf = 1 / (1 + np.exp((V + 68.9) / 6.5))
        tau_S = np.exp((V + 183.6) / 15.24)
        tau_F = np.exp((V + 158.6) / 11.2) / (1 + np.exp((V + 75) / 5.5))

        alpha_S = h_inf / tau_S
        beta_S = (1 - h_inf) / tau_S
        alpha_F = h_inf / tau_F
        beta_F = (1 - h_inf) / tau_F

        dS1dt = alpha_S * (1 - S1 - S2) - beta_S * S1
        dS2dt = -k2 * (S2 - C * S1)  # k2 and C are constants to be defined
        dF1dt = alpha_F * (1 - F1 - F2) - beta_F * F1
        dF2dt = -k2 * (F2 - C * F1)

        # Update S1, S2, F1, F2 outside this method

        return self.g_max * (S1 + S2) * (F1 + F2) * (V - self.E_rev)

