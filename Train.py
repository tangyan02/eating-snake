import random

import numpy as np
import torch
from tqdm import tqdm

from Game import GameEnvironment
from DQN import DQN
from ReplayBuffer import ReplayBuffer


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode = num_episodes / 10 * i + i_episode + 1
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值

                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if episode % 1000 == 0:
                    agent.save(f"model/model_{episode}.mdl")

                if episode % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % episode,
                        'return':
                            '%.3f' % np.mean(return_list[-100:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list


lr = 1e-5
num_episodes = 10000
gamma = 0.90
epsilon = 0.03
target_update = 50  # 将目标网络更新到当前价值网络锁需的步数间隔

buffer_size = 5000  # replay buffer 保存的样本总数
minimal_size = 1000  # replay buffer 达到这个数量，才开始取样学习
batch_size = 64  # replay buffer 每次随机采样样本数
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = GameEnvironment(14, 0, -1, 1)
action_dim = 4

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
agent = DQN(action_dim, lr, gamma, epsilon, target_update, device, "DoubleDQN")
return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)
