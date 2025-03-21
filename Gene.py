import random
import numpy as np
class GeneticAlgorithm:
    def __init__(self, network, pop_size=30, crossover_prob=0.2, mutation_prob=0.2, generations=800,a=0.05):
        self.network = network
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.population = []
        self.end=0
        self.a=a


    def initialize_population(self, start, end):
        for _ in range(self.pop_size):
            path = self.random_path(start, end)
            self.population.append(path)

    def random_path(self, start, end):
        graph = {node: [] for node in range(1, self.network.num_nodes + 1)}
        for arc in self.network.arcs:
            graph[arc[0]].append(arc[1])

        def dfs(node, path, visited):
            if node == end:
                return path
            visited.add(node)
            next_nodes = graph[node]
            random.shuffle(next_nodes)
            for next_node in next_nodes:
                if next_node not in visited:
                    new_path = dfs(next_node, path + [(node, next_node)], visited.copy())
                    if new_path:
                        return new_path
            return []

        return dfs(start, [], set())


    def rank_based_evaluation(self):
        sorted_population = sorted(self.population, key=lambda path: self.fitness(path), reverse=True)
        a = self.a
        rank_evaluation = []
        for i in range(len(sorted_population)):
            rank_score = a * (1 - a) ** i
            rank_evaluation.append((sorted_population[i], rank_score))
        return rank_evaluation


    def selection(self):
        ranked_population = self.rank_based_evaluation()
        total_rank_score = sum(rank_score for _, rank_score in ranked_population)
        probabilities = [rank_score / total_rank_score for _, rank_score in ranked_population]
        selected_index = np.random.choice(len(ranked_population), p=probabilities)
        return ranked_population[selected_index][0]

    def is_path_connected(self, path):
        """
        检查路径是否连通
        """
        if not path:
            return False
        for i in range(len(path) - 1):
            start, end = path[i]
            next_start, _ = path[i + 1]
            if end != next_start:
                return False
        if path[-1][1]!=self.end:
            return False
        return True

    def crossover(self, chromosome1, chromosome2):
        start_node1, end_node1 = chromosome1[0][0], chromosome1[-1][1]
        nodes_in_chromosome1 = {node for arc in chromosome1[1:-1] for node in arc if
                                node not in (start_node1, end_node1)}
        nodes_in_chromosome2 = {node for arc in chromosome2[1:-1] for node in arc if
                                node not in (start_node1, end_node1)}
        # 求两个集合的交集，得到公共节点
        common_nodes = nodes_in_chromosome1.intersection(nodes_in_chromosome2)
        if common_nodes:
            common_node = random.choice(list(common_nodes))
            index1 = next((i for i, arc in enumerate(chromosome1) if common_node == arc[1]), None)
            index2 = next((i for i, arc in enumerate(chromosome2) if common_node == arc[1]), None)
            new_chromosome1 = chromosome1[:index1 + 1] + chromosome2[index2 + 1:]
            new_chromosome2 = chromosome2[:index2 + 1] + chromosome1[index1 + 1:]
            if self.is_path_connected(new_chromosome1) and self.is_path_connected(new_chromosome2):
                return new_chromosome1, new_chromosome2
            else:
                print("Gene crossover error")
                print(chromosome1,chromosome2)
                print(new_chromosome1,new_chromosome2)
                print(index1,index2)
                print(common_node)
        else:
            return chromosome1, chromosome2


    def mutate(self, path):
        if random.random() < self.mutation_prob:
            index = random.randint(0, len(path) - 1)
            start_node = path[index][0]
            mutated_path = path[:index]
            mutated_path += self.random_path(start_node, self.end)
            return mutated_path
        return path


    def run(self, start, end, objective='mean', threshold=None, alpha=None, fitness_change_threshold=1e-5):
        self.initialize_population(start, end)
        current_generation=1
        add_fitness=0
        avg_fitness=0
        avg_new_fitness=0
        self.end=end
        fitness_change=float('inf')
        while current_generation < self.generations and fitness_change > fitness_change_threshold:
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = self.selection()
                parent2 = self.selection()
                if random.random() < self.crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = new_population
            if self.population:
                current_best_path = max(self.population,
                                        key=lambda path: self.fitness(path, objective, threshold, alpha))
                add_fitness += self.fitness(current_best_path, objective, threshold, alpha)
                avg_new_fitness=add_fitness/current_generation
                fitness_change=abs(avg_fitness-avg_new_fitness)
                avg_fitness=avg_new_fitness
            current_generation +=1
        # 确保传递所需参数
        print(f"算法收敛代数：{current_generation}")
        best_path = max(self.population, key=lambda path: self.fitness(path, objective, threshold, alpha))
        return best_path,current_generation


    def fitness(self, path, objective='mean', threshold=None, alpha=None):
        if objective == 'mean':
            mean_length, _ = self.network.simulate_path_length(path)
            return 1 / mean_length
        elif objective == 'probability' and threshold is not None:
            prob = self.network.calculate_probability(path, threshold)
            return prob
        elif objective == 'alpha' and alpha is not None:
            alpha_length = self.network.calculate_alpha_shortest(path, alpha)
            return 1 / alpha_length
        else:
            # 处理未定义的 objective 值，可以抛出异常或者返回一个默认值
            raise ValueError(f"未定义的 objective 值: {objective}")


