import GenerateNetwork
import StochasticNetwork
import Dijkstra
import Gene
import time
import pickle
network=GenerateNetwork.generate_random_network(num_nodes=10000, extra_arc_prob=0.1,limit=100)
GenerateNetwork.analyze_network(10000, network)

with open(f'SaveNet/network_{network["num_nodes"]}.pickle', 'wb') as savefile:
    pickle.dump(network, savefile)
'''
with open('network.pickle', 'rb') as f:
    loaded_network = pickle.load(f)
'''
GeneParams={
    "pop_size":30,
    "generations":100,
    "crossover":0.2,
    "mutation":0.2,
    "fitness_change_threshold":1e-6
}
threshold_list=[80,90,100]
alpha_list=[0.7,0.8,0.9]
def comparetest():
    net=StochasticNetwork.StochasticNetwork(network["num_nodes"],network["arcs"],network["arc_lengths"])
    di_ga = Dijkstra.GeneticAlgorithm(net, pop_size=GeneParams["pop_size"], generations=GeneParams["generations"],
                               crossover_prob=GeneParams["crossover"], mutation_prob=GeneParams["mutation"])
    gene_ga=Gene.GeneticAlgorithm(net, pop_size=GeneParams["pop_size"], generations=GeneParams["generations"],
                               crossover_prob=GeneParams["crossover"], mutation_prob=GeneParams["mutation"])

    print("begin_test_mean")
    test_mean(di_ga,net=net)
    test_threshold(di_ga,net=net)
    test_alpha(di_ga,net=net)
    test_mean(gene_ga,net=net)
    test_threshold(gene_ga,net=net)
    test_alpha(gene_ga,net=net)
    return

def test_mean(ga,net,end=network["num_nodes"]):
    start_time = time.time()
    best_path, generation = ga.run(1, end, objective='mean',fitness_change_threshold=GeneParams["fitness_change_threshold"])
    end_time = time.time()
    total_time = end_time - start_time
    mean_length, _ = net.simulate_path_length(best_path)
    print(f"最优路径 (期望路径长度最小化): {best_path}")
    print(f"该路径的期望长度: {mean_length:.2f}")
    print(f"算法收敛代数：{generation}")
    print(f"总运行时间: {total_time} 秒")
    return


def test_threshold(ga,net,threshold=8000,end=network["num_nodes"]):
    start_time = time.time()
    best_path, generation = ga.run(1, end, objective='probability', threshold=threshold,fitness_change_threshold=GeneParams["fitness_change_threshold"])
    end_time = time.time()
    total_time = end_time - start_time
    probability = net.calculate_probability(best_path, threshold)
    print(f"最优路径 (概率路径优化，阈值={threshold}): {best_path}")
    print(f"该路径小于阈值 {threshold} 的概率: {probability:.2%}")
    print(f"算法收敛代数：{generation}")
    print(f"总运行时间: {total_time} 秒")
    return

def test_alpha(ga,net,alpha=0.9,end=network["num_nodes"]):
    start_time = time.time()
    best_path_alpha,generation = ga.run(1, end, objective='alpha', alpha=alpha,fitness_change_threshold=GeneParams["fitness_change_threshold"])
    end_time = time.time()
    total_time = end_time - start_time
    alpha_length = net.calculate_alpha_shortest(best_path_alpha, alpha)
    print(f"最优路径 (α-最短路径，置信度={alpha}): {best_path_alpha}")
    print(f"该路径在置信度 {alpha} 下的最短长度: {alpha_length:.2f}")
    print(f"算法收敛代数：{generation}")
    print(f"总运行时间: {total_time} 秒")
    return


comparetest()