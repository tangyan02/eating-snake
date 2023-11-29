import time

import pygame
import numpy as np
from Game import GameEnvironment
import torch
from DQN import Qnet

gridsize = 14  # 13
framerate = 10
block_size = 20

qNet = Qnet()
qNet.load_state_dict(torch.load('model/model_10000.mdl'))

env = GameEnvironment(gridsize, 0., -1, 1)
env.reset()

windowwidth = gridsize * block_size * 2
windowheight = gridsize * block_size

pygame.init()  # pygame 初始化
win = pygame.display.set_mode((windowwidth, windowheight))  # 设置pygame窗口
pygame.display.set_caption("snake")
font = pygame.font.SysFont('arial', 18)
clock = pygame.time.Clock()


def drawboard(snake, apple):  # 通过pygame绘制可视化贪吃蛇运动的
    win.fill((0, 0, 0))
    for pos in snake.prevpos:  # 逐个绘制贪吃蛇的身体（贪吃蛇身体由不同的小block组成）
        pygame.draw.rect(win, (0, 255, 0), (pos[0] * block_size, pos[1] * block_size, block_size, block_size))
    pygame.draw.rect(win, (255, 0, 0),
                     (apple.pos[0] * block_size, apple.pos[1] * block_size, block_size, block_size))  # 绘制苹果


def pause():
    paused = True
    while paused == True:
        clock.tick(10)
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                paused = False


runGame = True
totalReward = 0
while runGame:
    clock.tick(framerate)

    state_0 = env.getState()

    state = torch.tensor(np.array(state_0), dtype=torch.float)
    action = qNet(state).argmax().item()

    next_state, reward, done, _ = env.step(action)

    drawboard(env.snake, env.apple)

    totalReward += reward
    lensnaketext = font.render('snake length: ' + str(env.snake.len + 1), False, (255, 255, 255))
    rewardtext = font.render('reword: ' + str(totalReward), False, (255, 255, 255))
    snakedirdtext = font.render('snake direction: ' + str(env.snake.getDirDesc()), False, (255, 255, 255))

    win.blit(lensnaketext, (windowwidth // 2, 40))
    win.blit(rewardtext, (windowwidth // 2, 80))
    win.blit(snakedirdtext, (windowwidth // 2, 160))

    for event in pygame.event.get():  # pygame 推出
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            runGame = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        pause()

    pygame.display.update()

    if done:
        time.sleep(1)
        env.reset()
        totalReward = 0

pygame.quit()
