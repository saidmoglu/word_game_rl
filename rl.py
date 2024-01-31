import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline

from GridWorldEnv import *
from time import sleep
from gymnasium.wrappers import FlattenObservation

env = GridWorldEnv()
env.render_mode = 'ansi'
_, info_total = env.reset()
print("\t".join(list(info_total)))
env = FlattenObservation(env)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from dqn_agent import Agent

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0) # , word_list=env.words, max_word_len=env.max_word_len)

# # watch an untrained agent
# state, _= env.reset()
# print(state)
# for j in range(50):
#     action = agent.act(state)
#     env.render()
#     state, reward, done, _, _ = env.step(action)
#     if done:
#         break 
        
# env.reset()


def dqn(n_episodes=500000, max_t=50, eps_start=1.0, eps_end=0.0001, eps_decay=0.9995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    episodes_interval = 100
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=episodes_interval)  # last episodes_interval scores
    eps = eps_start                    # initialize epsilon
    valid_words_count = 0
    best_score = -1000
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                valid_words_count += 1
                break 
        for x in info_total:
            info_total[x] += float(info[x])/episodes_interval
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}'.format(i_episode), end="")
        if i_episode % episodes_interval == 0:
            info_values = "\t".join(["{:.2f}".format(x) for x in info_total.values()])
            avg_score = np.mean(scores_window)
            print('\rEpisode {}\tAvg Score: {:.2f}\teps:{:.4f}\tvalid words:{}\t{}'.format(i_episode, avg_score,eps,valid_words_count, info_values))
            valid_words_count = 0
            for x in info_total:
                info_total[x] = 0
            if(avg_score > best_score):
                best_score = avg_score
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
