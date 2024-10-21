# Skrypt do trenowania strategii żeglarza w postaci tablicy Q 

import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf
from tqdm import tqdm

number_of_episodes = 10000                   # number of training epizodes (multi-stage processes) 
gamma = 1                                    # discount factor

file_name = 'map_simple.txt'
# file_name = 'map_easy.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained action-value table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

strategy = np.random.randint(low=1,high=5,size=np.shape(reward_map))  # random strategy
random_strategy_mean_reward = np.mean(sf.sailor_test(reward_map,strategy,1000))
sf.draw_strategy(reward_map,strategy,"random_strategy_average_reward_=_" + str(np.round(random_strategy_mean_reward,2)))
alphas = []
epsilons = []
for episode in tqdm(range(number_of_episodes)):
    alpha = sf.alpha_linear_decay(episode, number_of_episodes, num_of_rows * num_of_columns)
    # if episode % 100 == 0:
    #     print(alpha)
    epsilon = sf.epsilon_linear_decay(episode, number_of_episodes, num_of_rows * num_of_columns)
    # if episode % 100 == 0:
    #     print(epsilon)

    alphas.append(alpha)
    epsilons.append(epsilon)

    state = np.zeros([2], dtype=int)
    state[0] = np.random.randint(0, num_of_rows)
    step = 0
    while not (step == num_of_steps_max) | sf.on_finish_line(state[1], num_of_columns):
        step += 1
        action = sf.choose_action_epsilon_greedy(state, Q, epsilon)
        state_next, reward = sf.environment(state, action, reward_map)
        best_next_action = np.argmax(Q[state_next[0], state_next[1], :]) + 1
        current_state_and_action = (state[0], state[1], action - 1)
        next_state_and_best_action = (state_next[0], state_next[1], best_next_action - 1)
        Q[current_state_and_action] += alpha * (
            reward + gamma * Q[next_state_and_best_action] - Q[current_state_and_action]
        )
        state = state_next

strategy = sf.strategy(Q)

plt.plot(alphas)
plt.plot(epsilons)
plt.legend(["aplha", "epsilon"])
plt.show()

sf.sailor_test(reward_map, strategy, 1000)
sf.draw_strategy(reward_map, strategy, "best_strategy")
