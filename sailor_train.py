import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 1000                   # number of training epizodes (multi-stage processes) 
gamma = 0.7                                 # discount factor
delta_max = 0.0001                            # threshold for convergence


file_name = 'map_small.txt'
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
sf.draw(reward_map,strategy,"random_strategy mean reward = " + str(random_strategy_mean_reward))

def value_iteration(reward_map, gamma, delta_max):
    V = np.zeros((num_of_rows, num_of_columns))  # Initialize value function V(s)
    delta = float('inf')  # Set delta to infinity to start the iteration
    
    while delta >= delta_max:
        V_pom = np.copy(V)  # Copy V to V_pom
        delta = 0  # Reset delta
        
        # Iterate over all states in the state space
        for i in range(num_of_rows):
            for j in range(num_of_columns):
                state = (i, j)
                # Update V(s) based on maximum expected future reward
                V[state] = max(
                    sum(p * (reward + gamma * V_pom[new_state])
                        for (new_state, p), reward in sf.get_transitions(state, action, reward_map))
                    for action in range(1, 5)
                )
                # Calculate the max difference in value for convergence check
                delta = max(delta, abs(V[state] - V_pom[state]))
                print(delta - delta_max)

    # Extract the optimal policy π(s) based on the learned value function V(s)
    strategy = np.zeros((num_of_rows, num_of_columns), dtype=int)
    for i in range(num_of_rows):
        for j in range(num_of_columns):
            state = (i, j)
            strategy[state] = np.argmax([
                sum(p * (reward + gamma * V[new_state])
                    for (new_state, p), reward in sf.get_transitions(state, action, reward_map))
                for action in range(1, 5)
            ]) + 1

    return strategy

# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................
optimal_strategy = value_iteration(reward_map, gamma=gamma, delta_max=delta_max)

sf.sailor_test(reward_map, optimal_strategy, 1000)
sf.draw(reward_map, optimal_strategy, "best_strategy")
