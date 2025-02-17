import itertools
import math
from collections import defaultdict
import heapq

# 定义不同分布的期望计算函数
def expected_time_normal(mean, std_dev):
    return mean

def expected_time_uniform(low, high):
    return (low + high) / 2

def expected_time_triangular(low, mode, high):
    return (low + mode + high) / 3

# 计算弧的期望旅行时间
def arc_expected_time(arc_info):
    dist_type = arc_info[0]
    if dist_type == 'N':
        mean, std_dev = arc_info[1:]
        return expected_time_normal(mean, std_dev)
    elif dist_type == 'U':
        low, high = arc_info[1:]
        return expected_time_uniform(low, high)
    elif dist_type == 'T':
        low, mode, high = arc_info[1:]
        return expected_time_triangular(low, mode, high)

# 生成所有可能的路径
def generate_all_paths(arcs, origin, destination):
    graph = defaultdict(list)
    for u, v in arcs:
        graph[u].append(v)
    paths = []
    def dfs(node, path):
        path = path + [node]
        if node == destination:
            paths.append(path)
        for neighbor in graph[node]:
            if neighbor not in path:
                dfs(neighbor, path)
    dfs(origin, [])
    return paths

# 计算路径的期望旅行时间
def path_expected_time(path, arc_lengths):
    total_time = 0
    for i in range(len(path) - 1):
        arc = (path[i], path[i + 1])
        total_time += arc_expected_time(arc_lengths[arc])
    return total_time

# 寻找最小期望旅行时间路径
def find_least_expected_travel_time_path(arcs, arc_lengths, origin, destination):
    all_paths = generate_all_paths(arcs, origin, destination)
    min_expected_time = float('inf')
    optimal_path = None
    for path in all_paths:
        expected_time = path_expected_time(path, arc_lengths)
        if expected_time < min_expected_time:
            min_expected_time = expected_time
            optimal_path = path
    return optimal_path, min_expected_time

# 示例网络数据
arcs = [(1, 2), (2, 5), (1, 3), (3, 5), (1, 4), (4, 5)]
arc_lengths = {
    (1, 2): ('N', 12, 3),
    (2, 5): ('U', 8, 12),
    (1, 3): ('N', 10, 1),
    (3, 5): ('T', 6, 9, 12),
    (1, 4): ('N', 15, 2),
    (4, 5): ('U', 5, 10)
}
origin = 1
destination = 5

# 调用函数寻找最优路径
optimal_path, min_expected_time = find_least_expected_travel_time_path(arcs, arc_lengths, origin, destination)
print(f"最优路径: {optimal_path}")
print(f"最小期望旅行时间: {min_expected_time}")