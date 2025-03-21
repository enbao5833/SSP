import StochasticNetwork
import numpy as np
import random

class QSSP:
    def __init__(self, network, source, destination, episode_num=1000, epsilon=0.1, alpha_0=0.25, gamma=0.95):
        self.network = network
        self.source = source
        self.destination = destination
        self.episode_num = episode_num
        self.epsilon = epsilon
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.num_nodes = network.num_nodes

        # 初始化 Q 表、QCounter 表和 QMeanReward 表
        self.Q = np.zeros((self.num_nodes + 1, self.num_nodes + 1))
        self.QCounter = np.zeros((self.num_nodes + 1, self.num_nodes + 1))
        self.QMeanReward = np.zeros((self.num_nodes + 1, self.num_nodes + 1))

    def epsilon_greedy(self, state, available_actions):
        if random.random() < self.epsilon:
            # Exploration: 随机选择一个可用动作
            return random.choice(available_actions)
        else:
            # Exploitation: 选择 Q 值最大的动作
            q_values = [self.Q[state][action] for action in available_actions]
            max_q_index = np.argmax(q_values)
            return available_actions[max_q_index]

    def step(self, state, action):
        arc = (state, action)
        reward = -self.network.get_random_length(arc)  # 奖励为边权的负值
        next_state = action
        return reward, next_state

    def run(self):
        path_set = []
        for episode in range(self.episode_num):
            state = self.source
            path = [state]
            while state != self.destination:
                # 获取当前节点的所有出边邻居节点
                action_set = [neighbor for neighbor in range(1, self.num_nodes + 1) if (state, neighbor) in self.network.arcs]
                # 排除已访问过的节点
                available_actions = [action for action in action_set if action not in path]
                if not available_actions:
                    break  # 死胡同，终止当前路径
                action = self.epsilon_greedy(state, available_actions)
                reward, next_state = self.step(state, action)
                total_reward = self.QMeanReward[state][action] * self.QCounter[state][action] + reward
                total_num = self.QCounter[state][action] + 1
                r_avg = total_reward / total_num
                self.QMeanReward[state][action] = r_avg
                self.QCounter[state][action] = total_num
                # 下一状态的最大 Q 值
                next_action_set = [neighbor for neighbor in range(1, self.num_nodes + 1) if (next_state, neighbor) in self.network.arcs]
                max_next_Q = max([self.Q[next_state][a] for a in next_action_set], default=0)
                # 衰减学习率
                alpha = self.alpha_0 / (1 + episode)
                # Q-learning 更新规则
                self.Q[state][action] += alpha * (r_avg + self.gamma * max_next_Q - self.Q[state][action])
                state = next_state
                path.append(state)
            path_set.append(path)
        # 生成最终路径
        final_path = [self.source]
        current_state = self.source
        while current_state != self.destination:
            action_set = [neighbor for neighbor in range(1, self.num_nodes + 1) if (current_state, neighbor) in self.network.arcs]
            q_values = [self.Q[current_state][action] for action in action_set]
            max_q_index = np.argmax(q_values)
            next_state = action_set[max_q_index]
            final_path.append(next_state)
            current_state = next_state
        return final_path


# 示例数据
arcs = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (4, 7), (4, 11), (5, 8), (5, 11), (5, 12), (6, 9),
        (6, 10),
        (7, 10), (7, 11), (8, 12), (8, 13), (9, 16), (10, 16), (10, 17), (11, 14), (11, 17), (12, 14), (12, 15),
        (13, 15),
        (13, 19), (14, 21), (15, 18), (15, 19), (16, 20), (17, 20), (17, 21), (18, 21), (18, 22), (18, 23),
        (19, 22),
        (20, 23), (21, 23), (22, 23)]

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

network = StochasticNetwork.StochasticNetwork(num_nodes=23, arcs=arcs, arc_lengths=arc_lengths)
qss = QSSP(network, source=1, destination=23)
final_path = qss.run()
print("最终路径:", final_path)