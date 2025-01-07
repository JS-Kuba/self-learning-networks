# static map from lecture:
type_of_map = -1          
obs_size = 3          # size of observable area e.g 3x3
if_cross = True       # observable area is cross-shaped e.g agent see only vertical and horizontal neighbouring cells

# other static map:
# type_of_map = -2          
# obs_size = 3          # size of observable area e.g 3x3
# if_cross = False      # squared observable area 

# # random map: middle:
# type_of_map = 0          
# obs_size = 3
# if_cross = False

# # random map: middle:
# type_of_map = 2          
# obs_size = 5
# if_cross = False

# # random map hard:
# type_of_map = 3          
# obs_size = 7
# if_cross = False

import numpy as np
import pdb
import animat_fun as afun
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

GAMMA = 0.99
LEARNING_RATE = 1e-3

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-3):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        # Deeper network with more hidden layers
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        temperature = 1.0
        adjusted_probs = probs.pow(1/temperature)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        highest_prob_action = np.random.choice(
            self.num_actions, 
            p=np.squeeze(adjusted_probs.detach().numpy())
        )
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []
    # compute discounted rewards
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

def my_action(strategy, observation):
    action, log_prob = strategy.get_action(observation)
    return action, log_prob

def animat_train(type_of_map, obs_size=3, if_cross=False, num_episodes=200):
    possible_actions = 4
    strategy = PolicyNetwork(obs_size**2, possible_actions, hidden_size=512, learning_rate=1e-3)

    gamma = 0.97
    map = afun.generate_map(type_of_map)
    num_of_rows, num_of_columns = np.shape(map)
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for epi in range(num_episodes):
        position = afun.start_position(map)
        max_num_of_steps = 10000
        
        if_end = False
        step_number = 0
        log_probs = []
        rewards = []

        while not if_end:
            step_number += 1
            
            observation = afun.observable_region(map, obs_size, position, if_cross)
            observation = observation.flatten()
            action, log_prob = my_action(strategy, observation)
            new_position, reward = afun.transition_and_reward(map, position, action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if (reward > 0) or (step_number > max_num_of_steps):
                if_end = True
                update_policy(strategy, rewards, log_probs)
                numsteps.append(step_number)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                
                if epi % 10 == 0:
                    sys.stdout.write(
                        "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(
                            epi,
                            np.round(np.sum(rewards), decimals=3),
                            np.round(np.mean(all_rewards[-10:]), decimals=3),
                            step_number
                        )
                    )

            position = new_position

    return strategy

def animat_test(strategy, type_of_map, obs_size=3, if_cross=False, num_episodes=100):
    gamma = 0.97
    mean_sum_of_discounted_rewards = 0
    total_episodes_with_positive_reward = 0

    for epi in range(num_episodes):
        map = afun.generate_map(type_of_map)
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4*(num_of_rows + num_of_columns)
        
        if_end = False
        step_number = 0
        sum_of_discounted_rewards = 0
        cumulated_gamma = 1
        path = []

        while not if_end:
            step_number += 1
            
            observation = afun.observable_region(map, obs_size, position, if_cross)
            observation = observation.flatten()
            action, log_prob = my_action(strategy, observation)

            new_position, reward = afun.transition_and_reward(map, position, action)

            path.append([*position, action, *new_position, reward])

            if (reward > 0) or (step_number > max_num_of_steps):
                if_end = True
                if reward > 0:
                    total_episodes_with_positive_reward += 1

            position = new_position
            sum_of_discounted_rewards += reward*cumulated_gamma
            cumulated_gamma *= gamma

        mean_sum_of_discounted_rewards += sum_of_discounted_rewards/num_episodes

        print(f"episode {epi}: steps = {step_number} sum_of_rewards = {sum_of_discounted_rewards}")

        if epi < 5:
            afun.save_map_and_path(map, path, epi)
            afun.save_text_animation(map, path, epi)

    print(f"after {num_episodes} episodes:")
    print(f"mean sum of discounted rewards = {mean_sum_of_discounted_rewards}")
    print(f"episodes with positive reward = {total_episodes_with_positive_reward}/{num_episodes}")

strategy = animat_train(type_of_map, obs_size, if_cross, num_episodes=300)
animat_test(strategy, type_of_map, obs_size, if_cross, num_episodes=100)

