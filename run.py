import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from Game import GameEnvironment
from model import QNetwork, get_network_input
from replay_buffer import ReplayMemory

epsilon = 0.05
epsilon_rate = 0.9999

gridSize = 14
GAMMA = 0.9
model = QNetwork()

epStart = 1750
if epStart > 0:
    model.load_state_dict(torch.load(f'model/model_{epStart}.ln'))
    epsilon = epsilon * pow(epsilon_rate, epStart)
    epStart += 1

board = GameEnvironment(gridSize, nothing=0, dead=-1, apple=1)
memory = ReplayMemory(1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def run_episode(num_games):
    global epsilon
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []

    while run:
        state = get_network_input(board)
        action_0 = model(state)
        rand = np.random.uniform(0, 1)

        if rand > epsilon:
            action = torch.argmax(action_0)
        else:
            action = np.random.randint(0, 5)

        ## update_boardstate the same snake till
        reward, done, len_of_snake = board.update_boardstate(action)
        next_state = get_network_input(board)

        memory.push(state, action, reward, next_state, done)

        total_reward += reward

        episode_games += 1

        if board.game_over == True:
            games_played += 1
            len_array.append(len_of_snake)
            board.resetgame()

            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake


MSE = nn.MSELoss()


def learn(num_updates, batch_size):
    total_loss = 0

    for i in range(num_updates):
        optimizer.zero_grad()
        sample = memory.sample(batch_size)

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states])
        dones = torch.FloatTensor(dones)

        q_local = model.forward(states)
        next_q_value = model.forward(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)

        Q_targets_next = torch.max(next_q_value, 1)[0] * (torch.ones(dones.size()) - dones)

        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)

        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss


# num_episodes = 60000
num_episodes = 10000
num_updates = 500
print_every = 10
games_in_episode = 30
batch_size = 20


def train():
    global epsilon
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    avg_len_array = []
    avg_max_len_array = []

    time_start = time.time()

    for i_episode in range(epStart, num_episodes + 1):

        epsilon = epsilon * epsilon_rate
        score, avg_len, max_len = run_episode(games_in_episode)
        scores_deque.append(score)
        scores_array.append(score)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        total_loss = learn(num_updates, batch_size)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print(
                'Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, Max.LenOfSnake:  {:.2f} Time: {:02}:{:02}:{:02} '. \
                    format(i_episode, total_loss, score, avg_len, max_len, dt // 3600, dt % 3600 // 60, dt % 60))

        memory.truncate()

        if i_episode % 250 == 0 and i_episode > 0:
            #     torch.save(model.state_dict(), './dir_chk_len/Snake_{}'.format(i_episode))
            torch.save(model.state_dict(), f"model/model_{i_episode}.ln")

            # plt.plot(np.arange(1 + epStart, len(avg_len_array) + 1 + epStart), avg_len_array, label="Avg Len of Snake")
            # plt.plot(np.arange(1 + epStart, len(avg_max_len_array) + 1 + epStart), avg_max_len_array,
            #          label="Max Len of Snake")
            # plt.legend(bbox_to_anchor=(1.05, 1))
            # plt.ylabel('Length of Snake')
            # plt.xlabel('Episodes #')
            # plt.show()


if __name__ == "__main__":
    train()
