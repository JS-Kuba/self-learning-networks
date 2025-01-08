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
import random

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

GAMMA = 0.99
LEARNING_RATE = 1e-3

class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-3):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions  # Add this line to store the number of actions
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
        action = np.random.choice(self.num_actions, p=probs.detach().numpy().squeeze())
        log_prob = torch.log(probs.squeeze(0)[action])
        return action, log_prob



class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, learning_rate=1e-3):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.network(state)


def update_actor_critic(actor, critic, rewards, log_probs, states):
    discounted_rewards = []
    Gt = 0
    for reward in reversed(rewards):
        Gt = reward + GAMMA * Gt
        discounted_rewards.insert(0, Gt)
    discounted_rewards = torch.tensor(discounted_rewards)
    
    # Normalize rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    states = torch.FloatTensor(states)
    values = critic(states).squeeze()
    
    advantages = discounted_rewards - values.detach()
    
    # Update actor
    actor_loss = (-torch.stack(log_probs) * advantages).mean()
    actor.optimizer.zero_grad()
    actor_loss.backward()
    actor.optimizer.step()
    
    # Update critic
    critic_loss = F.mse_loss(values, discounted_rewards)
    critic.optimizer.zero_grad()
    critic_loss.backward()
    critic.optimizer.step()


def animat_train_actor_critic(type_of_map, obs_size=3, if_cross=False, num_episodes=200):
    possible_actions = 4
    hidden_size = 512
    actor = ActorNetwork(obs_size**2, possible_actions, hidden_size, learning_rate=1e-3)
    critic = CriticNetwork(obs_size**2, hidden_size, learning_rate=1e-3)

    map = afun.generate_map(type_of_map)
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
        states = []

        while not if_end:
            step_number += 1
            
            observation = afun.observable_region(map, obs_size, position, if_cross)
            observation = observation.flatten()
            action, log_prob = actor.get_action(observation)
            new_position, reward = afun.transition_and_reward(map, position, action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(observation)

            if (reward > 0) or (step_number > max_num_of_steps):
                if_end = True
                update_actor_critic(actor, critic, rewards, log_probs, states)
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

    return actor, critic

def my_action(strategy, observation):
    action, log_prob = strategy.get_action(observation)
    return action, log_prob

def animat_test(strategy, type_of_map, obs_size=3, if_cross=False, num_episodes=100):
    gamma = 0.97
    mean_sum_of_discounted_rewards = 0
    total_episodes_with_positive_reward = 0

    positive_outcome_episodes = []  # Store episodes with positive outcome as (reward, map, path)

    for epi in range(num_episodes):
        map = afun.generate_map(type_of_map)
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4 * (num_of_rows + num_of_columns)

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
                    positive_outcome_episodes.append((sum_of_discounted_rewards, map, path))

            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

        mean_sum_of_discounted_rewards += sum_of_discounted_rewards / num_episodes

        print(f"episode {epi}: steps = {step_number} sum_of_rewards = {sum_of_discounted_rewards}")

    # Randomly select 10 episodes from those with positive outcome
    selected_episodes = random.sample(positive_outcome_episodes, min(10, len(positive_outcome_episodes)))

    # Save the selected episodes
    for rank, (reward, map, path) in enumerate(selected_episodes):
        afun.save_map_and_path(map, path, rank)
        afun.save_text_animation(map, path, rank)

    print(f"after {num_episodes} episodes:")
    print(f"mean sum of discounted rewards = {mean_sum_of_discounted_rewards}")
    print(f"episodes with positive reward = {total_episodes_with_positive_reward}/{num_episodes}")


strategy_actor, strategy_critic = animat_train_actor_critic(type_of_map, obs_size, if_cross, num_episodes=300)
animat_test(strategy_actor, type_of_map, obs_size, if_cross, num_episodes=100)
