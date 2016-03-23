import numpy as np


class LIFNeuron:

    def __init__(self, identity, input_connections, input_weights,output_connections, base_potential=0,
                 potential_threshold=0.5, refractory_time=6, leak_rate=0.02):
        self.identity = identity
        self.input_connections = input_connections
        self.input_weights=np.array(input_weights)
        self.output_connections = output_connections
        self.potential_threshold = potential_threshold
        self.refractory_time = refractory_time
        self.current_time = 0
        self.base_potential = base_potential
        self.potential = self.base_potential
        self.leak_rate = leak_rate
        self.is_refractory = False
        self.has_spiked = False
        self.refractoriness = 0
        self.spike_history = []

    def update_potential(self, spike):
        idx = list(np.nonzero(spike))
        self.potential += sum(self.input_weights[idx])

    def reset_potential(self):
        self.potential = self.base_potential

    def update_state(self):
        if self.potential > self.potential_threshold:
            self.reset_potential()
            self.has_spiked = True
            self.spike_history.append(self.current_time)
            self.is_refractory = True
            self.refractoriness = self.refractory_time

    def simulate_neuron(self, spike):
        if not self.is_refractory:
            spike = np.array(spike)
            current_spike = spike[:, -1]
            self.update_potential(current_spike)
            self.update_state()
        else:
            self.refractoriness = max(0, self.refractoriness-1)
            if self.refractoriness <= 0:
                self.is_refractory = False
        self.current_time += 1
        return self.spike_history, self.potential

    @staticmethod
    def stdp_learning(pre_spike, post_spike, learning_rate, positive_spike_importance, negative_spike_importance):
        pre_spike_energy = 0
        post_spike_energy = 0
        time_of_spike = pre_spike.shape[0]

        # calculate LTD
        if pre_spike[-1] == 1:
            post_spike_index = np.squeeze(np.array(np.nonzero(post_spike)))
            for i in range(0, post_spike_index.shape[0]):
                post_spike_energy += learning_rate*np.exp(-(1-negative_spike_importance)*(time_of_spike -
                                                                                          post_spike_index[i]))

        # calculate LTP
        if post_spike[-1] == 1:
            pre_spike_index = np.squeeze(np.array(np.nonzero(pre_spike)))
            for i in range(0, pre_spike_index.shape[0]):
                pre_spike_energy += learning_rate*np.exp(-(1-positive_spike_importance)*(time_of_spike-pre_spike_index[i]))

        return pre_spike_energy-post_spike_energy







