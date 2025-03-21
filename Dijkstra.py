import random
import numpy as np
class GeneticAlgorithm:
    def __init__(self, network, pop_size=30, crossover_prob=0.2, mutation_prob=0.2, generations=800, a=0.05):
        self.network = network
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.population = []
        self.end = 0
        self.a = a
        self.arcgraph = self.build_graph()

    def build_graph(self):
        # 将每个节点对应的字典值初始化为集合
        graph = {node: set() for node in range(1, self.network.num_nodes + 1)}
        for arc in self.network.arcs:
            graph[arc[0]].add(arc[1])
        return graph


    def initialize_population(self, start, end):
        for _ in range(self.pop_size):
            path = self.random_dijkstra(start, end)
            if path:
                self.population.append(path)
        if not self.population:
            print("警告：初始化种群为空！")


    def random_dijkstra(self, start, end):
        fixed_arc_lengths = {arc: self.network.get_random_length(arc) for arc in self.network.arcs}
        graph = {node: {} for node in range(1, self.network.num_nodes + 1)}
        for arc in self.network.arcs:
            if arc[0] not in graph:
                graph[arc[0]] = {}
            graph[arc[0]][arc[1]] = fixed_arc_lengths[arc]

        distances = {node: float('inf') for node in range(1, self.network.num_nodes + 1)}
        distances[start] = 0
        previous = {node: None for node in range(1, self.network.num_nodes + 1)}
        unvisited = set(range(1, self.network.num_nodes + 1))
        path = []

        # 初始时直接将起始节点作为当前节点
        current = start
        while unvisited:
            if current == end:
                while current:
                    if previous[current]:
                        path.insert(0, (previous[current], current))
                    current = previous[current]
                return path

            unvisited.remove(current)

            for neighbor, weight in graph[current].items():
                alt = distances[current] + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current

            # 后续循环中再通过概率选择当前节点
            current = None
            probs = []
            valid_nodes = []
            for node in unvisited:
                if distances[node] != float('inf'):
                    probs.append(distances[node])
                    valid_nodes.append(node)
            if not probs:
                break
            total = sum(probs)
            probs = [prob / total for prob in probs]
            current = random.choices(valid_nodes, weights=probs)[0]

        return path


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
        if not ranked_population:
            print("警告：排名后的种群为空！")
            return []
        total_rank_score = sum(rank_score for _, rank_score in ranked_population)
        probabilities = [rank_score / total_rank_score for _, rank_score in ranked_population]
        # 检查概率总和是否接近 1
        if abs(sum(probabilities) - 1) > 1e-6:
            print(f"警告：概率总和不为 1，总和为 {sum(probabilities)}")
        selected_index = np.random.choice(len(ranked_population), p=probabilities)
        return ranked_population[selected_index][0]

    def crossover(self, chromosome1, chromosome2):
        # 找出两个父代染色体中的公共节点
        # 找出两个父代染色体中的公共节点，去掉起始和结束节点
        start_node1, end_node1 = chromosome1[0][0], chromosome1[-1][1]
        nodes_in_chromosome1 = {node for arc in chromosome1[1:-1] for node in arc if
                                node not in (start_node1, end_node1)}
        nodes_in_chromosome2 = {node for arc in chromosome2[1:-1] for node in arc if
                                node not in (start_node1, end_node1)}
        # 求两个集合的交集，得到公共节点
        common_nodes = nodes_in_chromosome1.intersection(nodes_in_chromosome2)
        if common_nodes:
            best_gene_pair = None
            best_fitness = float('-inf')
            for node in common_nodes:
                index1 = next((i for i, arc in enumerate(chromosome1) if node == arc[1]), None)
                index2 = next((i for i, arc in enumerate(chromosome2) if node == arc[1]), None)
                if index1 is not None and index2 is not None:
                    new_chromosome1 = chromosome1[:index1 + 1] + [arc for arc in chromosome2[index2 + 1:]]
                    new_chromosome2 = chromosome2[:index2 + 1] + [arc for arc in chromosome1[index1 + 1:]]
                    if self.is_path_connected(new_chromosome1) and self.is_path_connected(new_chromosome2):
                        fitness1 = self.fitness(new_chromosome1)
                        fitness2 = self.fitness(new_chromosome2)
                        if fitness1 > best_fitness:
                            best_fitness=fitness1
                            best_gene_pair=(index1,index2)
                        if fitness2 > best_fitness:
                            best_fitness=fitness2
                            best_gene_pair=(index1,index2)
            if best_gene_pair:
                index1, index2 = best_gene_pair
                new_chromosome1 = chromosome1[:index1 + 1] + [arc for arc in chromosome2[index2 + 1:]]
                new_chromosome2 = chromosome2[:index2 + 1] + [arc for arc in chromosome1[index1 + 1:]]
                if new_chromosome1[-1][1]!=self.end or new_chromosome2[-1][1]!=self.end:
                    print("path error")
                    print(new_chromosome1,new_chromosome2)
                    print(best_gene_pair)
                    print(chromosome1,chromosome2)
                    print(common_nodes)
                return new_chromosome1, new_chromosome2
        return chromosome1, chromosome2

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

    def mutate(self, path):
        if random.random() < self.mutation_prob:
            index = random.randint(0, len(path) - 1)
            start_node = path[index][0]
            mutated_path = path[:index]
            mutated_path += self.random_dijkstra(start_node, self.end)
            if mutated_path[-1][1] != self.end:
                print("mutate error")
            return mutated_path
        return path


    def run(self, start, end, objective='mean', threshold=None, alpha=None, fitness_change_threshold=1e-5):
        self.end=end
        self.initialize_population(start, end)
        current_generation = 1
        add_fitness = 0
        avg_fitness = 0
        avg_new_fitness = 0
        fitness_change = float('inf')
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
                avg_new_fitness = add_fitness / current_generation
                fitness_change = abs(avg_fitness - avg_new_fitness)
                avg_fitness = avg_new_fitness
            current_generation += 1
        # 确保传递所需参数
        if self.population:
            best_path = max(self.population, key=lambda path: self.fitness(path, objective, threshold, alpha))
            return best_path,current_generation
        else:
            print("警告：最终种群为空！")
            return []


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