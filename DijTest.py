
import Dijkstra
import time
# 定义测试函数
import StochasticNetwork
import numpy as np

# 定义不同网络参数的列表
network_params_list = [
    {
        "num_nodes": 23,
        "arcs": [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (4, 7), (4, 11), (5, 8), (5, 11), (5, 12), (6, 9),
                 (6, 10), (7, 10), (7, 11), (8, 12), (8, 13), (9, 16), (10, 16), (10, 17), (11, 14), (11, 17), (12, 14),
                 (12, 15), (13, 15), (13, 19), (14, 21), (15, 18), (15, 19), (16, 20), (17, 20), (17, 21), (18, 21),
                 (18, 22), (18, 23), (19, 22), (20, 23), (21, 23), (22, 23)],
        "arc_lengths": {
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
    },
    # 你可以在这里添加更多的网络参数
]

# 定义不同遗传算法参数的列表
ga_params_list = [
    {
        "pop_size": 20,
        "generations": 100,
        "crossover_prob": 0.2,
        "mutation_prob": 0.2
    },
    # 你可以在这里添加更多的遗传算法参数
]

# 定义不同Dijkstra算法参数的列表（这里假设Dijkstra算法有一些参数，例如阈值）
dijkstra_params_list = [
    {
        "threshold": 65
    },
    {
        "threshold": 60
    },
    {
        "threshold": 55
    },
    {
        "threshold": 50
    }
]


def DjiTest():
    '''
    for network_params in network_params_list:
        num_nodes = network_params["num_nodes"]
        arcs = network_params["arcs"]
        arc_lengths = network_params["arc_lengths"]

        # 创建网络对象
        network = StochasticNetwork.StochasticNetwork(num_nodes=num_nodes, arcs=arcs, arc_lengths=arc_lengths)

        for ga_params in ga_params_list:
            pop_size = ga_params["pop_size"]
            generations = ga_params["generations"]
            crossover_prob = ga_params["crossover_prob"]
            mutation_prob = ga_params["mutation_prob"]

            # 创建遗传算法对象
            ga = Dijkstra.GeneticAlgorithm(network, pop_size=pop_size, generations=generations,
                                       crossover_prob=crossover_prob, mutation_prob=mutation_prob)

            for dijkstra_params in dijkstra_params_list:
                threshold = dijkstra_params["threshold"]

                print(f"测试网络参数: num_nodes={num_nodes}, ga参数: pop_size={pop_size}, generations={generations}, "
                      f"crossover_prob={crossover_prob}, mutation_prob={mutation_prob}, "
                      f"Dijkstra参数: threshold={threshold}")

                # 运行算法得到最优路径（概率路径优化）
                start_time = time.time()
                best_prob_path = ga.run(1, num_nodes, objective='probability', threshold=threshold)
                end_time = time.time()
                total_time = end_time - start_time
                probability = network.calculate_probability(best_prob_path, threshold)
                print(f"最优路径 (概率路径优化，阈值={threshold}): {best_prob_path}")
                print(f"该路径小于阈值 {threshold} 的概率: {probability:.2%}")
                print(f"总运行时间 (概率路径优化): {total_time} 秒")

            # 运行算法得到最优路径（期望路径长度最小化）
            start_time = time.time()
            best_path_mean = ga.run(1, num_nodes, objective='mean')
            end_time = time.time()
            total_time = end_time - start_time
            mean_length, _ = network.simulate_path_length(best_path_mean)
            print(f"最优路径 (期望路径长度最小化): {best_path_mean}")
            print(f"该路径的期望长度: {mean_length:.2f}")
            print(f"总运行时间 (期望路径长度最小化): {total_time} 秒")

            # 运行算法得到最优路径（α-最短路径）
            alpha = 0.9
            start_time = time.time()
            best_path_alpha = ga.run(1, num_nodes, objective='alpha', alpha=alpha)
            end_time = time.time()
            total_time = end_time - start_time
            alpha_length = network.calculate_alpha_shortest(best_path_alpha, alpha)
            print(f"最优路径 (α-最短路径，置信度={alpha}): {best_path_alpha}")
            print(f"该路径在置信度 {alpha} 下的最短长度: {alpha_length:.2f}")
            print(f"总运行时间 (α-最短路径): {total_time} 秒")
'''
    num_nodes = network_params_list[0]["num_nodes"]
    arcs = network_params_list[0]["arcs"]
    arc_lengths = network_params_list[0]["arc_lengths"]

    # 创建网络对象
    network = StochasticNetwork.StochasticNetwork(num_nodes=num_nodes, arcs=arcs, arc_lengths=arc_lengths)
    ga = Dijkstra.GeneticAlgorithm(network, pop_size=30, generations=150,
                               crossover_prob=0.3, mutation_prob=0.2)
    generation_list = []
    for i in range(1, 80):
        start_time = time.time()
        best_path, generation = ga.run(1, num_nodes, objective='probability',threshold=50)
        end_time = time.time()
        total_time = end_time - start_time
        probability = network.calculate_probability(best_path, 50)
        print(f"最优路径 (概率路径优化，阈值={50}): {best_path}")
        print(f"该路径小于阈值 {50} 的概率: {probability:.2%}")
        print(f"算法收敛代数：{generation}")
        # print(f"总运行时间 (期望路径长度最小化): {total_time} 秒")
        generation_list.append(generation)
        print(generation_list)
    print(np.mean(generation_list))

if __name__ == "__main__":
    DjiTest()
