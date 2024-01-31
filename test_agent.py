import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline

from GridWorldEnv import *
from time import sleep
from gymnasium.wrappers import FlattenObservation

env = GridWorldEnv()
env.render_mode = 'human'
env.reset()
env = FlattenObservation(env)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from dqn_agent import Agent

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)


# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(10):
    state, _ = env.reset()
    for j in range(50):
        action = agent.act(state)
        env.render()
        state, reward, done, _, _ = env.step(action)
        if done:
            break 