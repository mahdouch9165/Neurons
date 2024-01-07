from .current import Current

class LeakyCurrent(Current):
    def __init__(self, e, g, dimensions, loop):
        super().__init__(e, g, dimensions, loop)

    def compute_current(self, V_m, time_index):
        # Vectorized computation of the leaky current
        self.current_matrix[..., time_index] = self.g * (self.e - V_m)
        return self.current_matrix[..., time_index]
