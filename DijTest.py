import StochasticNetwork
import Dijkstra

arcs = [(1, 2), (2, 5), (1, 3), (3, 5), (1, 4), (4, 5)]
arc_lengths = {
    (1, 2): ('N', 12, 3),
    (2, 5): ('U', 8, 12),
    (1, 3): ('N', 10, 1),
    (3, 5): ('T', 6, 9, 12),
    (1, 4): ('N', 15, 2),
    (4, 5): ('U', 5, 10)
}

network = StochasticNetwork.StochasticNetwork(num_nodes=5, arcs=arcs, arc_lengths=arc_lengths)
ga = Dijkstra.GeneticAlgorithm(network, pop_size=50, generations=1000, crossover_prob=0.7, mutation_prob=0.4)

# 概率路径优化测试
threshold = 20
best_prob_path = ga.run(1, 5, objective='probability', threshold=threshold)
probability = network.calculate_probability(best_prob_path, threshold)

print(f"最优路径 (概率路径优化，阈值={threshold}): {best_prob_path}")
print(f"该路径小于阈值 {threshold} 的概率: {probability:.2%}")