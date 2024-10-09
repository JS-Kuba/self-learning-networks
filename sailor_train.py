# Skrypt do trenowania strategii żeglarza w postaci tablicy Q 

import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
from tqdm import tqdm

number_of_episodes = 800                   # number of training epizodes (multi-stage processes) 
gamma = 1                                 # discount factor


# file_name = 'map_simple.txt'
# file_name = 'map_easy.txt'
file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained action-value table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

strategy = np.random.randint(low=1,high=5,size=np.shape(reward_map))  # random strategy
random_strategy_mean_reward = np.mean(sf.sailor_test(reward_map,strategy,1000))
sf.draw_strategy(reward_map,strategy,"random_strategy_average_reward_=_" + str(np.round(random_strategy_mean_reward,2)))


# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
actions = [1, 2, 3, 4]
state_action_pairs = []
for x in range(num_of_rows):
    for y in range(num_of_columns):
        for a in actions:
            state_action_pairs.append(((x, y), a))

## Strategy iteration (episodes = 1000, gamma = 0.8)
# strategies_equal = False
# while not strategies_equal:
#     strategy_temp = strategy.copy()
#     for pair in tqdm(state_action_pairs):
#         for episode in range(number_of_episodes):
#             step = 0
#             state = np.array(pair[0])
#             action = pair[1]
#             finish = False
#             while not finish:
#                 step += 1
#                 state_new, reward = sf.environment(state, action, reward_map)
#                 state = state_new
#                 action = strategy[state[0], state[1]]
#                 if (state[1] >= num_of_columns - 1) | (step >= num_of_steps_max):
#                     finish = True
#                 sum_of_rewards[episode] += gamma*reward
            
#         Q[pair[0][0], pair[0][1], pair[1] - 1] = np.mean(sum_of_rewards)

#         sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

#     for x in range(num_of_rows):
#         for y in range(num_of_columns):
#             if y < num_of_columns - 1:
#                 strategy[x, y] = np.argmax(Q[x, y]) + 1
#             else:
#                 strategy[x, y] = 0
    

#     print("Testing strategy")
#     iter_rewards = np.mean(sf.sailor_test(reward_map, strategy, 1000))
#     if np.array_equal(strategy, strategy_temp):
#         strategies_equal = True
#     if iter_rewards >= 7:
#         strategies_equal = True
#         sf.draw_strategy(reward_map,strategy,"strategy_best_=_" + str(np.round(iter_rewards,2)))
###

## Value iteration (episodes = 800, gamma = 1)
alpha = lambda n: 1/(n+1)
for episode in tqdm(range(number_of_episodes), desc="Episodes"):
    for pair in state_action_pairs:
        step = 0
        state = np.array(pair[0])
        action = pair[1]
        rewards = []
        finish = False
        while not finish:
            step += 1
            state_next, reward = sf.environment(state, action, reward_map)
            rewards.append(reward)
            state = state_next
            action = np.argmax(Q[state[0], state[1]]) + 1
            if (state[1] >= num_of_columns - 1) | (step >= num_of_steps_max):
                finish = True

        rewards_sum = sum([gamma**i * r for i, r in enumerate(rewards)])
        Q[pair[0][0], pair[0][1], pair[1] - 1] = \
            (1-alpha(episode)) * \
            Q[pair[0][0], pair[0][1], pair[1] - 1] + \
            alpha(episode)*rewards_sum
            
for x in range(num_of_rows):
    for y in range(num_of_columns):
        if y < num_of_columns - 1:
            strategy[x, y] = np.argmax(Q[x, y]) + 1
        else:
            strategy[x, y] = 0

sf.sailor_test(reward_map, strategy, 1000)
sf.draw_strategy(reward_map,strategy,"strategy_best_iter_big")
###