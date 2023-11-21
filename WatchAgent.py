import pygame
import numpy as np
from Game import GameEnvironment
import torch
from DQN import Qnet

gridsize = 14  # 13
framerate = 10
block_size = 20

qNet = Qnet()
qNet.load_state_dict(torch.load('model/model_1000.mdl'))

env = GameEnvironment(gridsize, 0., -100., 100.)
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


runGame = True

while runGame:
    clock.tick(framerate)

    state_0 = env.getState()

    state = torch.tensor(np.array(state_0), dtype=torch.float)
    action = qNet(state).argmax().item()

    next_state, reward, done, _ = env.step(action)

    drawboard(env.snake, env.apple)

    lensnaketext = font.render(' length of snake: ' + str(env.snake.len + 1), False, (255, 255, 255))

    win.blit(lensnaketext, (windowwidth // 2, 40))

    for event in pygame.event.get():  # pygame 推出
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            runGame = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_r]:
        paused = True
        while paused == True:
            clock.tick(10)
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    paused = False

    pygame.display.update()

    if done == True:
        env.reset()

pygame.quit()