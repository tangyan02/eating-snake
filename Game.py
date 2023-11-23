import random

import numpy as np

# 定义贪吃蛇的动作，
# R表示向右边运动，就在在positon坐标的X轴加1,Y轴坐标不变，可以实现贪吃蛇头部坐标更新。
import torch

player_moves = {
    'L': np.array([-1., 0.]),
    'R': np.array([1., 0.]),
    'U': np.array([0., -1.]),
    'D': np.array([0., 1.])
}
initial_playersize = 4

dx = [0, 0, -1, 1]
dy = [1, -1, 0, 0]


class snakeclass(object):
    def __init__(self, gridsize):
        self.pos = np.array([gridsize // 2, gridsize // 2]).astype('float')  # 初始化贪吃蛇头部position坐标，位于显示区域的中心
        self.dir = np.array([1., 0.])  # 初始化贪吃蛇的运动
        self.len = initial_playersize  # 初始化贪吃蛇的长度
        # 建立一个position数据用于存贮贪吃蛇历史轨迹，这些轨迹也是贪吃蛇的身子
        self.prevpos = [np.array([gridsize // 2, gridsize // 2]).astype('float')]
        self.gridsize = gridsize

    def move(self):
        self.pos += self.dir
        self.prevpos.append(self.pos.copy())
        self.prevpos = self.prevpos[-self.len - 1:]

    def checkdead(self, pos):  # 判断贪吃蛇的头部是否到达边界或则碰到自己的身体。
        if pos[0] <= -1 or pos[0] >= self.gridsize:
            return True
        elif pos[1] <= -1 or pos[1] >= self.gridsize:
            return True
        # 判断贪吃蛇头是否碰到了身体
        elif list(pos) in [list(item) for item in self.prevpos[:-1]]:
            return True
        else:
            return False

    def getDirDesc(self):
        for k, v in player_moves.items():
            if (v == self.dir).all():
                return k

    def __len__(self):
        return self.len + 1


class appleclass(object):  # 定义苹果出现的地方
    def __init__(self, gridSize, snake):
        self.snake = snake
        self.gridSize = gridSize
        self.pos = np.random.randint(1, gridSize, 2)
        self.reset()

    def reset(self):
        count = 0
        inBody = [[0] * self.gridSize for _ in range(self.gridSize)]
        for pos in self.snake.prevpos:
            inBody[round(pos[0])][round(pos[1])] = 1

        for i in range(0, self.gridSize):
            for j in range(0, self.gridSize):
                if inBody[i][j] == 0:
                    count += 1

        k = random.randint(1, count)
        count = 0
        for i in range(0, self.gridSize):
            for j in range(0, self.gridSize):
                if inBody[i][j] == 0:
                    count += 1
                    if count == k:
                        self.pos = np.array([i, j])
                        return


class GameEnvironment(object):
    def __init__(self, gridsize, nothing, dead, apple):
        self.snake = snakeclass(gridsize)
        self.apple = appleclass(gridsize, self.snake)
        self.game_over = False
        self.gridsize = gridsize
        self.reward_nothing = nothing
        self.reward_dead = dead
        self.reward_apple = apple
        self.time_since_apple = 0

    def step(self, action):
        reward, done = self.update_boardstate(action)
        state = self.getState()
        return state, reward, done, None

    def reset(self):
        self.apple.pos = np.random.randint(1, self.gridsize, 2).astype('float')  # 随机初始化苹果的位置
        self.apple.score = 0  # 初始化score
        self.snake.pos = np.random.randint(1, self.gridsize, 2).astype('float')  # 初始化贪吃蛇出现的位置
        self.snake.prevpos = [self.snake.pos.copy().astype('float')]
        self.snake.len = initial_playersize
        self.game_over = False
        return self.getState()

    def getState(self):
        snake = self.snake
        apple = self.apple

        size = self.gridsize + 2

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

        state = np.stack([channel_1, channel_2, channel_3, channel_4], axis=0)
        return state

    def tryFill(self, x, y, map):
        count = 0
        for i in range(4):
            xx = int(x) + dx[i]
            yy = int(y) + dy[i]

            if xx < 0 or xx >= self.gridsize or yy < 0 or yy >= self.gridsize:
                continue
            if map[xx][yy] == 0:
                map[xx][yy] = 1
                count += 1
                count += self.tryFill(xx, yy, map)
        return count

    def isSpaceEnough(self):
        map = [[0] * self.gridsize for _ in range(self.gridsize)]
        snakeBodyCount = 0
        for pos in self.snake.prevpos:
            if pos[0] < 0 or pos[0] >= self.gridsize or pos[1] < 0 or pos[1] >= self.gridsize:
                return 0
            map[round(pos[0])][round(pos[1])] = 1
            snakeBodyCount += 1
        spaceCount = self.tryFill(int(self.snake.pos[0]), int(self.snake.pos[1]), map)
        return snakeBodyCount + spaceCount == self.gridsize * self.gridsize

    def update_boardstate(self, move):
        reward = self.reward_nothing
        Done = False
        # python 中的all函数用于判断可迭代iterable中所有元素是否都是True，如果是返回Treu，否则False
        if move == 0:
            if not (self.snake.dir == player_moves['R']).all():
                self.snake.dir = player_moves['L']
        if move == 1:
            if not (self.snake.dir == player_moves['L']).all():
                self.snake.dir = player_moves['R']
        if move == 2:
            if not (self.snake.dir == player_moves['D']).all():
                self.snake.dir = player_moves['U']
        if move == 3:
            if not (self.snake.dir == player_moves['U']).all():
                self.snake.dir = player_moves['D']
        self.snake.move()
        self.time_since_apple += 1
        # --
        if self.time_since_apple == 200:  # 到达一定步数没吃到苹果，则认为失败
            self.game_over = True
            reward = self.reward_dead
            self.time_since_apple = 0
            Done = True
        # --
        if self.snake.checkdead(self.snake.pos) == True:  # 碰到边缘和身子，结束游戏
            self.game_over = True
            reward = self.reward_dead
            self.time_since_apple = 0
            Done = True
        elif (self.snake.pos == self.apple.pos).all():  # 判断是否吃到苹果
            self.apple.reset()
            self.snake.len += 1
            self.time_since_apple = 0
            reward = self.reward_apple

        return reward, Done
