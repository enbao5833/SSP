import StochasticNetwork
import Gene
import time
# 定义测试函数
def test_random_dijkstra():
    arcs = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (4, 7), (4, 11), (5, 8), (5, 11), (5, 12), (6, 9),
            (6, 10),
            (7, 10), (7, 11), (8, 12), (8, 13), (9, 16), (10, 16), (10, 17), (11, 14), (11, 17), (12, 14), (12, 15),
            (13, 15),
            (13, 19), (14, 21), (15, 18), (15, 19), (16, 20), (17, 20), (17, 21), (18, 21), (18, 22), (18, 23),
            (19, 22),
            (20, 23), (21, 23), (22, 23)]

    # 定义弧长分布函数
    arc_lengths = {
        (1, 2): ('N', 12, 1),
        (1, 3): ('N', 9, 1),
        (1, 4): ('T', 8, 10, 12),
        (1, 5): ('N', 7, 1),
        (2, 6): ('T', 5, 7, 15),
        (2, 7): ('U', 6, 11),
        (3, 8): ('U', 10, 16),
        (4, 7): ('EXP', 20),
        (4, 11): ('T', 6, 10, 13),
        (5, 11): ('U', 7, 13),
        (5, 12): ('EXP', 13),
        (5, 8): ('T', 6, 9, 10),
        (6, 9): ('T', 6, 8, 10),
        (6, 10): ('T', 10, 11, 14),
        (7, 10): ('U', 9, 12),
        (7, 11): ('U', 6, 8),
        (8, 12): ('U', 5, 9),
        (8, 13): ('N', 5, 1),
        (9, 16): ('EXP', 7),
        (10, 16): ('U', 12, 16),
        (10, 17): ('T', 15, 17, 19),
        (11, 17): ('EXP', 9),
        (11, 14): ('N', 9, 1),
        (12, 14): ('T', 10, 13, 15),
        (12, 15): ('N', 12, 2),
        (13, 15): ('T', 10, 12, 14),
        (13, 19): ('T', 17, 18, 19),
        (14, 21): ('N', 11, 1),
        (15, 18): ('T', 8, 9, 11),
        (15, 19): ('N', 7, 1),
        (16, 20): ('T', 9, 10, 12),
        (17, 20): ('T', 7, 11, 12),
        (17, 21): ('U', 6, 8),
        (18, 21): ('N', 15, 2),
        (18, 23): ('T', 5, 7, 9),
        (18, 22): ('EXP', 5),
        (19, 22): ('U', 15, 17),
        (20, 23): ('T', 13, 14, 15),
        (21, 23): ('T', 12, 13, 15),
        (22, 23): ('U', 4, 6)
    }
    # 创建网络对象
    network = StochasticNetwork.StochasticNetwork(num_nodes=23, arcs=arcs, arc_lengths=arc_lengths)
    # 创建遗传算法对象
    ga = Gene.GeneticAlgorithm(network, pop_size=30, generations=100, crossover_prob=0.2, mutation_prob=0.2)
    threshold = 90
    # 运行算法得到最优路径
    start_time = time.time()
    best_prob_path = ga.run(1, 23, objective='probability', threshold=threshold)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总运行时间: {total_time} 秒")
    probability = network.calculate_probability(best_prob_path, threshold)
    print(f"最优路径 (概率路径优化，阈值={threshold}): {best_prob_path}")
    print(f"该路径小于阈值 {threshold} 的概率: {probability:.2%}")
    start_time = time.time()
    best_path_mean = ga.run(1, 23, objective='mean')
    end_time = time.time()
    total_time = end_time - start_time
    mean_length, _ = network.simulate_path_length(best_path_mean)
    print(f"最优路径 (期望路径长度最小化): {best_path_mean}")
    print(f"该路径的期望长度: {mean_length:.2f}")
    print(f"总运行时间: {total_time} 秒")
    alpha = 0.9
    start_time = time.time()
    best_path_alpha = ga.run(1, 23, objective='alpha', alpha=alpha)
    end_time = time.time()
    total_time = end_time - start_time
    alpha_length = network.calculate_alpha_shortest(best_path_alpha, alpha)
    print(f"最优路径 (α-最短路径，置信度={alpha}): {best_path_alpha}")
    print(f"该路径在置信度 {alpha} 下的最短长度: {alpha_length:.2f}")
    print(f"总运行时间: {total_time} 秒")
if __name__ == "__main__":
    test_random_dijkstra()
