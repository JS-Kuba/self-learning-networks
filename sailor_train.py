# Skrypt do trenowania strategii żeglarza w postaci tablicy Q 

import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 1000                   # number of training epizodes (multi-stage processes) 
gamma = 1                                # discount factor
delta_max = 0.0001                            # threshold for convergence

# file_name = 'map_simple.txt'
#file_name = 'map_easy.txt'
file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained action-value table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

strategy = np.random.randint(low=1,high=5,size=np.shape(reward_map))  # random strategy
random_strategy_mean_reward = np.mean(sf.sailor_test(reward_map,strategy,1000))
sf.draw_strategy(reward_map,strategy,"random_strategy_average_reward_=_" + str(np.round(random_strategy_mean_reward,2)))


# Dynamic Programming - strategy iteration
while True:
    delta = delta_max
    V = np.zeros([num_of_rows, num_of_columns], dtype=float)
    strategy_temp = strategy.copy()

    while delta >= delta_max:
        V_temp = V.copy()
        delta = 0

        for i in range(num_of_rows):
            for j in range(num_of_columns):
                if j < num_of_columns - 1:
                    action = strategy[i, j]
                    state = np.array([i, j])
                    transitions = sf.get_transitions(state, action, reward_map)
                    mean_reward = sf.get_mean_reward(transitions)

                    V[i, j] = mean_reward + gamma * sum(
                        p * V_temp[new_state[0], new_state[1]] for (new_state, p), reward in transitions
                    )
                    delta = max(delta, abs(V[i, j] - V_temp[i, j]))

    for i in range(num_of_rows):
        for j in range(num_of_columns):
            if j < num_of_columns - 1:
                action_values = np.zeros(4)
                for action in range(1, 5):
                    state = np.array([i, j])
                    transitions = sf.get_transitions(state, action, reward_map)
                    mean_reward = sf.get_mean_reward(transitions)
                    
                    action_values[action - 1] = mean_reward + gamma * sum(p * V[new_state[0], new_state[1]] for (new_state, p), reward in transitions)
                strategy[i, j] = np.argmax(action_values) + 1
            else:
                strategy[i, j] = 0

    if np.array_equal(strategy, strategy_temp):
        print("finish")
        break

sf.sailor_test(reward_map, strategy, 1000)
sf.draw_strategy(reward_map, strategy, "best_strategy")
