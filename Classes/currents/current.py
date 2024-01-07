import numpy as np

class Current:
    def __init__(self, e, g, dimensions, loop):
        self.e = e
        self.g = g
        self.current_matrix = np.zeros(dimensions + (loop,))  # Time dimension added

    def compute_current(self, V_m, time_index):
        raise NotImplementedError("Subclasses should implement this method")
