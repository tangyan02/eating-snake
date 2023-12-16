import os
import random
import time

import numpy as np
import torch

from Game import GameEnvironment
from PPO import PPO


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
        agent.save(f"model/actor_new.mdl", f"model/critic_new.mdl")

    if i_episode % 10000 == 0:
        print(f"{getTimeStr()}开始保存模型，episode:{i_episode}")
        agent.save(f"model/actor_{i_episode}.mdl", f"model/critic_{i_episode}.mdl")
        print(f"{getTimeStr()}模型保存完毕")
        f = open(f"logs/returns.log", "w")
        f.write(str.join(" ", map(str, return_list)))
        f.close()
        print(f"{getTimeStr()}日志写入完毕")


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(1, num_episodes + 1):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        doAfterPerEpisde(agent, i, return_list)
        return_list.append(episode_return)
        agent.update(transition_dict)
    return return_list


dirPreBuild()

lr = 1e-5
num_episodes = 100000

actor_lr = 1e-4
critic_lr = 5e-3
gamma = 0.95
lmbda = 0.95
epochs = 10
eps = 0.2

device = getDevice()
env = GameEnvironment(14, 0, -1, 1)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

agent = PPO(actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

train_on_policy_agent(env, agent, num_episodes)
