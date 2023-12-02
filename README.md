# eating-snake

## 简介
14*14的地图的贪吃蛇游戏，使用DQN算法训练

## 依赖
```
pip install pygame
pip install numpy
pip intsall torch
```
## 文件介绍
WatchAgent.py：启动游戏观看AI自己玩

Train.py: 从0开始训练模型

model: 目录中存放已经训练好的模型，目前有50000次episode

### 训练效果
探索率代表每个时间步，随机action的概率
|探索率|收敛平均长度|
|  ----  | ----  |
|0.1|30|
|0.01|35|

