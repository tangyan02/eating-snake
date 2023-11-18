import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=4 * 4 * 32, out_features=512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=5)

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
        x = self.fc2(x)
        return x


def get_network_input(board):
    snake = board.snake
    apple = board.apple

    size = 16

    channel_1 = np.zeros((size, size))
    channel_1[int(snake.pos[0] + 1)][int(snake.pos[1] + 1)] = 1

    channel_2 = np.zeros((size, size))
    channel_2[int(apple.pos[0]) + 1][int(apple.pos[1] + 1)] = 1

    channel_3 = np.zeros((size, size))
    for pos in snake.prevpos:
        channel_3[int(pos[0] + 1)][int(pos[1] + 1)] = 1

    channel_4 = np.zeros((size, size))
    for i in range(size):
        channel_4[i][0] = 1
        channel_4[0][i] = 1
        channel_4[size - 1][i] = 1
        channel_4[i][size - 1] = 1

    input_tensor = np.stack([channel_1, channel_2, channel_3, channel_4], axis=0)
    return torch.from_numpy(input_tensor).float()
