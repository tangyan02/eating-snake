import os
import random
import time

import numpy as np
import torch

from Game import GameEnvironment
from DQN import DQN
from ReplayBuffer import ReplayBuffer


def getTimeStr():
    return time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime())


def dirPreBuild():
    dirs = ["model", "logs"]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)


def getDevice():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            device = "mps"
    print(f"{getTimeStr()}device:{device}")
    return device


def doAfterPerEpisde(agent, i_episode, return_list):
    if i_episode % 100 == 0:
        print(f"{getTimeStr()}episode:{'%d' % i_episode} return:{'%.3f' % np.mean(return_list[-100:])}")

    if i_episode % 1000 == 0:
        print(f"{getTimeStr()}开始保存模型，episode:{i_episode}")
        agent.save(f"model/model_{i_episode}.mdl")
        print(f"{getTimeStr()}模型保存完毕")
        f = open(f"logs/returns.log", "w")
        f.write(str.join(" ", map(str, return_list)))
        f.close()
        print(f"{getTimeStr()}日志写入完毕")


def train_DQN(agent, env, num_episodes, started_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i_episode in range(started_episodes + 1, num_episodes):
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

        doAfterPerEpisde(agent, i_episode, return_list)
    return return_list, max_q_value_list


dirPreBuild()

lr = 1e-5
num_episodes = 100000

gamma = 0.90
epsilon = 0.1
target_update = 200  # 将目标网络更新到当前价值网络锁需的步数间隔

buffer_size = 10000  # replay buffer 保存的样本总数
minimal_size = 1000  # replay buffer 达到这个数量，才开始取样学习
batch_size = 100  # replay buffer 每次随机采样样本数device = "cpu"

device = getDevice()
env = GameEnvironment(14, 0, -1, 1)
action_dim = 4

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
agent = DQN(action_dim, lr, gamma, epsilon, target_update, device, "DoubleDQN")

started_episodes = 0
if started_episodes > 0:
    agent.load(f"model/model_{started_episodes}.mdl")

return_list, max_q_value_list = train_DQN(agent, env, num_episodes, started_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)
