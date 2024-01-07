import numpy as np
from .neuronPatch import NeuronPatch

class Simulation:
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.loop_num = int(T / dt)
        self.patches = []

    def add_patch(self, patch):
        patch.dt = self.dt
        self.patches.append(patch)

    def run(self):
        for i in range(self.loop_num - 1):
            for patch in self.patches:
                patch.update_state(i)
                patch.update_transmitter_release(i)