import numpy as np
class StochasticNetwork:
    def __init__(self, num_nodes, arcs, arc_lengths):
        self.num_nodes = num_nodes
        self.arcs = arcs
        self.arc_lengths = arc_lengths

    def get_random_length(self, arc):
        dist = self.arc_lengths[arc]
        if dist[0] == 'N':
            mean, stddev = dist[1], dist[2]
            return np.random.normal(mean, stddev)
        elif dist[0] == 'U':
            low, high = dist[1], dist[2]
            return np.random.uniform(low, high)
        elif dist[0] == 'EXP':
            rate = dist[1]
            return np.random.exponential(rate)
        elif dist[0] == 'T':
            low, mode, high = dist[1], dist[2], dist[3]
            return np.random.triangular(low, mode, high)

    def simulate_path_length(self, path, num_simulations=5000):
        lengths = []
        for _ in range(num_simulations):
            length = sum(self.get_random_length(arc) for arc in path)
            lengths.append(length)
        return np.mean(lengths), lengths

    def calculate_probability(self, path, threshold, num_simulations=5000):
        if not path:
            return 0.0
        _, lengths = self.simulate_path_length(path, num_simulations)
        count = sum(1 for l in lengths if l <= threshold)
        prob = count / num_simulations
        return min(max(prob, 0.0), 1.0)

    def calculate_alpha_shortest(self, path, alpha, num_simulations=5000):
        _, lengths = self.simulate_path_length(path, num_simulations)
        sorted_lengths = sorted(lengths)
        index = int((1 - alpha) * num_simulations)
        return sorted_lengths[index]
