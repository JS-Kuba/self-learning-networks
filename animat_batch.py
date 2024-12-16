import numpy as np
import pdb
import animat_fun as afun
from tqdm import tqdm

import wandb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


# static map from lecture:
type_of_map = -1          
obs_size = 3          # size of observable area e.g 3x3
if_cross = True       # observable area is cross-shaped e.g agent see only vertical and horizontal neighbouring cells

# Hyperparameters
batch_size = 8  # Batch size for training
learning_rate = 0.01
gamma = 0.97   # Discount factor
number_of_episodes = 300
epsilon = 1
epsilon_decay = 0.99


# Create model
def create_policy_network(obs_size, action_size, dense_size):
    model = Sequential([
        Flatten(input_shape=(obs_size, obs_size)),  # Flatten observation (3x3 -> 9)
        Dense(dense_size, activation='relu'),       # Hidden layer
        Dense(dense_size, activation='relu'),
        Dense(action_size, activation='softmax')   # Output - action probabilities
    ])
    return model

# Training function
def train_policy(policy_network, optimizer, observations, actions, rewards, gamma=0.97, batch_size=32):
    # Calculate cumulative rewards (G_t) for each experience
    G = []
    discounted_sum = 0
    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        G.insert(0, discounted_sum)
    G = np.array(G)

    # Normalize rewards for stability
    G = (G - np.mean(G)) / (np.std(G) + 1e-8)
    
    # Convert lists to numpy arrays for batch processing
    observations = np.array(observations)
    actions = np.array(actions)
    
    # Create a data generator for batching
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions, G)).batch(batch_size)
    
    total_loss = 0
    for batch_obs, batch_actions, batch_G in dataset:
        with tf.GradientTape() as tape:
            logits = policy_network(batch_obs)
            neg_log_probs = tf.keras.losses.sparse_categorical_crossentropy(batch_actions, logits, from_logits=False)

            # Ensure both tensors are float32 for compatibility
            neg_log_probs = tf.cast(neg_log_probs, tf.float32)
            batch_G = tf.cast(batch_G, tf.float32)
            
            loss = tf.reduce_mean(neg_log_probs * batch_G)
        
        grads = tape.gradient(loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
        total_loss += loss.numpy()

    return total_loss / len(dataset)


# Initialize the policy network and optimizer
action_size = 4  # Number of actions (up, down, left, right)
dense_size = 32
policy_network = create_policy_network(obs_size, action_size, dense_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)


# Action selection (epsilon-greedy with exploration/exploitation)
def my_action(policy_network, observation, epsilon):
    observation = np.expand_dims(observation, axis=0)  # Add batch dimension
    action_probs = policy_network.predict(observation, verbose=0)[0]  # Predicted action probabilities
    if np.random.rand() < epsilon:  # Epsilon-greedy
        action = np.random.choice(action_size)
    else:
        action = np.random.choice(len(action_probs), p=action_probs)  # Choose action based on policy
    return action


# Main training function
def animat_train(type_of_map, obs_size=3, if_cross=False, log=True, batch_size=8):
    global epsilon  # Use the global epsilon value
    
    # Initialize experiment logging
    if log:
        wandb.init(project="zpd-project", config={
            "gamma": gamma,
            "episodes": number_of_episodes,
            "obs_size": obs_size,
            "if_cross": if_cross,
            "architecture": f"2xDense({dense_size})",
            "lr": learning_rate,
            "batch_size": batch_size
        })

    # Initialize experience storage
    observations, actions, rewards = [], [], []

    for epi in tqdm(range(number_of_episodes)):
        epsilon = max(0.01, epsilon * epsilon_decay / (epi / 1000 + 1))  # Decay epsilon
        
        map = afun.generate_map(type_of_map)  # Generate map
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4 * (num_of_rows + num_of_columns)

        if_end = False
        step_number = 0
        cumulated_gamma = 1
        sum_of_discounted_rewards = 0

        while not if_end:  # Episode steps loop
            step_number += 1

            # Observation and action selection
            observation = afun.observable_region(map, obs_size, position, if_cross)
            action = my_action(policy_network, observation, epsilon)

            # Transition to the new state
            new_position, reward = afun.transition_and_reward(map, position, action)

            # Store experience
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            # Update position and accumulate discounted rewards
            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

            # Check for episode termination
            if reward > 0 or step_number > max_num_of_steps:
                if_end = True

        # Train policy with the batch of experiences collected in the episode
        loss = train_policy(policy_network, optimizer, observations, actions, rewards, gamma, batch_size)

        if log:
            wandb.log({
                "total_reward": sum_of_discounted_rewards,
                "loss": loss,
                "epsilon": epsilon
            })

        # Clear experience buffer after each episode
        observations, actions, rewards = [], [], []

        print(f"\nEpisode {epi+1}: Total Reward = {sum_of_discounted_rewards:.2f}, Loss = {loss:.4f}")

    if log:
        wandb.finish()

    return policy_network  # Return trained policy network


# Testing function (unchanged)
def animat_test(strategy, type_of_map, obs_size=3, if_cross=False):
    gamma = 0.97  # Same gamma as in training
    number_of_episodes = 100

    mean_sum_of_discounted_rewards = 0

    for epi in range(number_of_episodes):
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
            action = my_action(strategy, observation, epsilon=0)  # Always exploit during testing

            new_position, reward = afun.transition_and_reward(map, position, action)
            path.append([*position, action, *new_position, reward])

            if reward > 0 or step_number > max_num_of_steps:
                if_end = True

            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

        mean_sum_of_discounted_rewards += sum_of_discounted_rewards / number_of_episodes
        print(f"Episode {epi+1}: Steps = {step_number}, Sum of Rewards = {sum_of_discounted_rewards}")

    print(f"After {number_of_episodes} episodes:")
    print(f"Mean Sum of Discounted Rewards = {mean_sum_of_discounted_rewards}")


# Run the training and testing
strategy = animat_train(type_of_map, obs_size, if_cross, log=True, batch_size=batch_size)
animat_test(strategy, type_of_map, obs_size, if_cross)
