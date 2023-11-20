import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=4 * 4 * 32, out_features=256)
        self.relu3 = nn.ReLU()
        self.fcA = nn.Linear(in_features=256, out_features=4)
        self.fcV = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        # 第一层卷积、激活函数和池化
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 第二层卷积、激活函数和池化
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 将数据平展成一维
        x = x.view(-1, 4 * 4 * 32)
        # 第一层全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        # 第二层全连接层
        A = self.fcA(x)
        V = self.fcV(x)
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


class DQN:
    ''' DQN算法,包括Double DQN '''

    def __init__(self,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):
        self.q_net = Qnet().to(device)
        self.target_q_net = Qnet().to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):

        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN':  # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:  # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
