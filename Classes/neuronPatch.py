import numpy as np

class NeuronPatch:
    def __init__(self, dimensions, V_initial, dt, loop):
        self.dimensions = dimensions
        self.dt = dt
        self.loop = loop
        self.V = np.full(dimensions + (loop,), V_initial, dtype=np.float64)
        self.receptors = []
        self.currents = []

    def add_receptor(self, receptor):
        self.receptors.append(receptor)

    def add_current(self, current):
        self.currents.append(current)

    def update_state(self, time_index):
        if time_index < 0 or time_index >= self.loop:
            raise IndexError("time_index out of bounds")

        total_current = np.zeros(self.dimensions)
        for current in self.currents:
            # Directly use the current without reshaping as it's already in the correct shape
            total_current += current.compute_current(self.V[..., time_index], time_index)

        # Update membrane potentials only if not the last time step
        if time_index < self.loop - 1:
            self.V[..., time_index + 1] = self.V[..., time_index] + self.dt * total_current

    def update_transmitter_release(self, time_index):
        for receptor in self.receptors:
            receptor.update_nt_release(self.V[..., time_index])
