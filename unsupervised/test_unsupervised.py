import unsupervised.neuron as nrn
import numpy as np
import matplotlib.pyplot as plt

number_of_neurons = 5
simulation_time = 1000
forward_connections = [[2, 3], [4], [1, 4], [2], [3]]
backward_connections = [[], [2], [0, 3], [0, 4], [1, 2]]
backward_weights = [[], [0.05], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05]]
history = 10

neurons = list()
neurons.append(nrn.LIFNeuron(2, backward_connections[2], backward_weights[2], forward_connections[2]))
# use list datastructure for spike. Does not effect neuron behaviour
spike = np.random.choice([0, 1], size=(len(backward_connections[2]), history))
print(neurons[0].stdp_learning(pre_spike=spike[0, :], post_spike=spike[1, :], learning_rate=0.1,
                               positive_spike_importance=0.7, negative_spike_importance=0.3))
# sim_potential = [0]
# for time in range(1, simulation_time):
#    (history, p) = neurons[0].simulate_neuron(spike)
#    sim_potential.append(p)
# plt.plot(sim_potential)
# plt.show()
