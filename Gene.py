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
        self.a=a


    def initialize_population(self, start, end):
        for _ in range(self.pop_size):
            path = self.random_path(start, end)
            self.population.append(path)


    def random_path(self, start, end):
        current = start
        path = []
        visited = set()
        while current!= end:
            visited.add(current)
            next_nodes = [arc[1] for arc in self.network.arcs if arc[0] == current]
            if not next_nodes:
                break
            next_node = random.choice(next_nodes)
            path.append((current, next_node))
            current = next_node
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
        total_rank_score = sum(rank_score for _, rank_score in ranked_population)
        probabilities = [rank_score / total_rank_score for _, rank_score in ranked_population]
        selected_index = np.random.choice(len(ranked_population), p=probabilities)
        return ranked_population[selected_index][0]


    def crossover(self, chromosome1, chromosome2):
        common_nodes = set(chromosome1).intersection(set(chromosome2))
        if common_nodes:
            common_node = random.choice(list(common_nodes))
            index1 = chromosome1.index(common_node)
            index2 = chromosome2.index(common_node)
            new_chromosome1 = chromosome1[:index1 + 1] + chromosome2[index2 + 1:]
            new_chromosome2 = chromosome2[:index2 + 1] + chromosome1[index1 + 1:]
            return new_chromosome1, new_chromosome2
        else:
            return chromosome1, chromosome2


    def mutate(self, path):
        if random.random() < self.mutation_prob:
            index = random.randint(0, len(path) - 1)
            start_node = path[index][0]
            mutated_path = path[:index]
            mutated_path += self.random_path(start_node, path[-1][1])
            return mutated_path
        return path


    def run(self, start, end, objective='mean', threshold=None, alpha=None):
        self.initialize_population(start, end)
        for generation in range(self.generations):
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
        # 确保传递所需参数
        best_path = max(self.population, key=lambda path: self.fitness(path, objective, threshold, alpha))
        return best_path


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

