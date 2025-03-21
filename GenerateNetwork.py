import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import pprint
def generate_random_network(num_nodes=100, extra_arc_prob=0.1,limit=20):
    # 生成基本的有向无环图（每个节点从2开始连接一个随机前驱）
    arcs = []
    for node in range(2, num_nodes+1):
        # 随机选取一个比 node 小的节点作为前驱
        if node > limit:
            potential_preds = list(range(node - limit, node))
        else:
            potential_preds = list(range(1, node))
        # 权重分配：节点编号越大，权重越高
        weights = [i**2 for i in potential_preds]
        # 进行加权随机选择
        pred = random.choices(potential_preds, weights=weights, k=1)[0]
        arcs.append((pred, node))

    # 在所有可能的 (i,j) (i < j) 中额外添加一些弧（避免重复）
    possible_arcs = [(i, j) for i in range(1, num_nodes) for j in range(i + 1, num_nodes + 1) if j - i <= limit]
    extra_arcs = [arc for arc in possible_arcs if arc not in arcs and random.random() < extra_arc_prob]
    arcs.extend(extra_arcs)

    # 定义可选的分布类型
    dist_types = ['N', 'T', 'U', 'EXP']

    # 为每个弧随机生成边长信息
    arc_lengths = {}
    for arc in arcs:
        dtype = random.choice(dist_types)
        if dtype == 'N':
            # 正态分布：生成 mean 和 std（整数），例如 mean ∈ [5, 15], std ∈ [1, 3]
            mean = random.randint(5, 20)
            std = random.randint(1, 10)
            params = (mean, std)
        elif dtype == 'T':
            # 三角分布：生成 left, mode, right，要求 left < mode < right
            left = random.randint(5, 10)
            mode = random.randint(left + 1, left + 5)
            right = random.randint(mode + 1, mode + 5)
            params = (left, mode, right)
        elif dtype == 'U':
            # 均匀分布：生成 low 和 high（low < high）
            low = random.randint(5, 20)
            high = random.randint(low + 3, low + 15)
            params = (low, high)
        elif dtype == 'EXP':
            # 指数分布：只生成一个参数（例如 scale），这里取 scale ∈ [10, 30]
            scale = random.randint(5, 30)
            params = (scale,)
        # 边长信息为一个元组，元组第一个元素是分布代码，后续为参数
        arc_lengths[arc] = (dtype,) + params

    # 返回生成的网络数据
    network = {
        "num_nodes": num_nodes,
        "arcs": arcs,
        "arc_lengths": arc_lengths
    }
    return network

def analyze_network(num_nodes, net):
    num_edges = len(net["arcs"])  # 计算边的数量
    density = num_edges / (num_nodes * (num_nodes - 1))  # 计算稠密度

    # 生成 NetworkX 有向图
    G = nx.DiGraph()
    G.add_nodes_from(range(1, num_nodes + 1))
    G.add_edges_from(net["arcs"])

    # 计算入度和出度
    in_degrees = [deg for _, deg in G.in_degree()]
    out_degrees = [deg for _, deg in G.out_degree()]

    # 计算统计信息
    degree_info = {
        "max_in_degree": max(in_degrees),
        "min_in_degree": min(in_degrees),
        "avg_in_degree": sum(in_degrees) / num_nodes,
        "max_out_degree": max(out_degrees),
        "min_out_degree": min(out_degrees),
        "avg_out_degree": sum(out_degrees) / num_nodes,
    }

    # 输出网络信息
    print("网络信息:")
    pprint.pprint(net)
    print(f"节点个数: {num_nodes}")
    print(f"边的条数: {num_edges}")
    print(f"网络稠密度: {density:.6f}")
    print(f"入度: max={degree_info['max_in_degree']}, min={degree_info['min_in_degree']}, avg={degree_info['avg_in_degree']:.2f}")
    print(f"出度: max={degree_info['max_out_degree']}, min={degree_info['min_out_degree']}, avg={degree_info['avg_out_degree']:.2f}")
    # 绘制网络图
    # 计算节点布局
    # 绘制节点
    '''
    plt.figure(figsize=(30, 30))
    pos = graphviz_layout(G, prog='dot')
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=10)
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=False)
    nx.draw_networkx_labels(G, pos)
    plt.title("Random Network Graph")
    # 显示图形
    plt.savefig(f'png/network_graph_{num_nodes}.png')
    # 关闭图形
    plt.close()
    '''
    return

