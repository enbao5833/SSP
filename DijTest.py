import unittest
import StochasticNetwork
import Dijkstra

import unittest
import StochasticNetwork
import Dijkstra


class TestRandomDijkstra(unittest.TestCase):

    def setUp(self):
        arcs = [(1, 2), (2, 5), (1, 3), (3, 5), (1, 4), (4, 5)]
        arc_lengths = {
            (1, 2): ('N', 12, 3),
            (2, 5): ('U', 8, 12),
            (1, 3): ('N', 10, 1),
            (3, 5): ('T', 6, 9, 12),
            (1, 4): ('N', 15, 2),
            (4, 5): ('U', 5, 10)
        }
        self.network = StochasticNetwork.StochasticNetwork(num_nodes=5, arcs=arcs, arc_lengths=arc_lengths)
        self.ga = Dijkstra.GeneticAlgorithm(self.network, pop_size=30, generations=100, crossover_prob=0.2,
                                            mutation_prob=0.2)
        self.threshold = 25
        best_prob_path = self.ga.run(1, 4, objective='probability', threshold=self.threshold)
        probability = self.network.calculate_probability(best_prob_path, self.threshold)

        print(f"最优路径 (概率路径优化，阈值={self.threshold}): {best_prob_path}")
        print(f"该路径小于阈值 {self.threshold} 的概率: {probability:.2%}")
