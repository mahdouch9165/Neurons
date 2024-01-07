from Classes.currents.potassium import PotassiumCurrent
from Classes.currents.sodium import SodiumCurrent
from Classes.currents.leaky import LeakyCurrent
from Classes.simulation import Simulation
from Classes.neuronPatch import NeuronPatch

if __name__ == "__main__":
    simulation = Simulation(T=700, dt=0.01)
    
    neuron_num = 10
    patch = NeuronPatch(N_num=neuron_num, V_initial=-65, dt=simulation.dt)
    patch.add_current(SodiumCurrent(50, 100, neuron_num, simulation.loop_num, 0.01))
    patch.add_current(PotassiumCurrent(-90, 10, neuron_num, simulation.loop_num, 0.01))
    patch.add_current(LeakyCurrent(-78, 0.05, neuron_num, simulation.loop_num))
    simulation.add_patch(patch)

    simulation.run()