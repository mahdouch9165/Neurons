from .current import Current

class LeakyCurrent(Current):
    def __init__(self, e, g, dimensions, loop):
        # Ensure e is negative (reversal potential) and g is positive (conductance)
        super().__init__(e, g, dimensions, loop)

    def compute_current(self, V_m, time_index):
        # Correctly calculate leaky current: I_leak = g * (V_m - e)
        self.current_matrix[..., time_index] = self.g * (self.e - V_m)
        return self.current_matrix[..., time_index]
