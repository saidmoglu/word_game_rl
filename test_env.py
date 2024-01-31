from GridWorldEnv import *
from time import sleep
from gymnasium.wrappers import FlattenObservation

env = GridWorldEnv()
env.render_mode = 'human'
env.reset()
# print('Number of actions: ', env.action_space.n)
# env = FlattenObservation(env)
# env.reset()
# print('State shape: ', env.observation_space.shape)
# env.render()
# input("press enter to exit")
# exit()
for i in range(200):
    action = env.action_space.sample()
    if(i%5 == 0):
        action = 8
    if(i%5 == 4):
        action = 9
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    print(reward)
    print("")
    env.render()
    # sleep(1)

input("press enter to exit")