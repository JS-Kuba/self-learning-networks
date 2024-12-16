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



# Tworzenie modelu sieci neuronowej
def create_policy_network(obs_size, action_size, dense_size):
    model = Sequential([
        Flatten(input_shape=(obs_size, obs_size)),  # Spłaszczenie obserwacji (3x3 -> 9)
        Dense(dense_size, activation='relu'),              # Warstwa ukryta
        Dense(dense_size, activation='relu'),
        Dense(action_size, activation='softmax')   # Wyjście - prawdopodobieństwa akcji
    ])
    return model

def train_policy(policy_network, optimizer, observations, actions, rewards, gamma=0.97):
    # Oblicz skumulowane nagrody (G_t)
    G = []
    discounted_sum = 0
    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        G.insert(0, discounted_sum)
    G = np.array(G)

    # Normalizacja nagród dla stabilności
    G = (G - np.mean(G)) / (np.std(G) + 1e-8)
    
    # Konwersja list na numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)

    # Gradient policy gradient
    with tf.GradientTape() as tape:
        logits = policy_network(observations)
        neg_log_probs = tf.keras.losses.sparse_categorical_crossentropy(actions, logits, from_logits=False)
        loss = tf.reduce_mean(neg_log_probs * G)

    grads = tape.gradient(loss, policy_network.trainable_variables)

    gradients_magnitude = [tf.norm(grad) for grad in grads]

    optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

    return loss.numpy(), gradients_magnitude

action_size = 4  # Liczba akcji (góra, dół, lewo, prawo)
dense_size = 32
policy_network = create_policy_network(obs_size, action_size, dense_size)
learning_rate=0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)

def my_action(policy_network, observation):
    observation = np.expand_dims(observation, axis=0)  # Dodanie wymiaru batch
    action_probs = policy_network.predict(observation, verbose=0)[0]  # Przewidywane prawdopodobieństwa
    action = np.random.choice(len(action_probs), p=action_probs)  # Losowy wybór z softmax
    return action

def animat_train(type_of_map, obs_size=3, if_cross=False, log=True):
    gamma = 0.97   # współczynnik dyskontowania
    number_of_episodes = 300
    epsilon = 1
    epsilon_decay = 0.99
    if log:
        wandb.init(project="zpd-project", config={
            "gamma": gamma,
            "episodes": number_of_episodes,
            "dense_size": dense_size,
            "lr": learning_rate
        })

    observations, actions, rewards = [], [], []  # Bufor na doświadczenia

    for epi in tqdm(range(number_of_episodes)):
        epsilon = max(0.01, epsilon * epsilon_decay/(epi/1000+1))
        map = afun.generate_map(type_of_map)  # Generacja mapy
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4 * (num_of_rows + num_of_columns)

        if_end = False
        step_number = 0
        cumulated_gamma = 1
        sum_of_discounted_rewards = 0

        while not if_end:  # Pętla kroków w epizodzie
            step_number += 1

            # Obserwacja i wybór akcji
            observation = afun.observable_region(map, obs_size, position, if_cross)

            if np.random.rand() < epsilon:  # epsilon-greedy
                action = np.random.choice(action_size)
            else:
                action = my_action(policy_network, observation)
            # Przejście do nowego stanu
            new_position, reward = afun.transition_and_reward(map, position, action)

            # Zbieranie danych
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            # Aktualizacja pozycji i sumowanie nagród
            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

            # Sprawdzenie warunków zakończenia
            if reward > 0 or step_number > max_num_of_steps:
                if_end = True

        loss, gradients_magnitude = train_policy(policy_network, optimizer, observations, actions, rewards)

        if log:
            wandb.log({
                "total_reward": sum_of_discounted_rewards,
                "loss": loss,
                "epslion": epsilon
            })

        # Czyszczenie buforów po epizodzie
        observations, actions, rewards = [], [], []

        print(f"\nEpisode {epi+1}: Total Reward = {sum_of_discounted_rewards:.2f}")
        if sum_of_discounted_rewards < 0.01 and sum_of_discounted_rewards > -0.01:
            print("Gradients magnitude: ", gradients_magnitude)


    return policy_network  # Zwróć model sieci jako strategię


def animat_test(strategy, type_of_map, obs_size = 3, if_cross = False):
    gamma = 0.97 # can be changed in training and test for the same value
    number_of_episodes = 10

    mean_sum_of_discounted_rewards = 0

    for epi in range(number_of_episodes):
        map = afun.generate_map(type_of_map)  # can be generate for more than one episode
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4*(num_of_rows + num_of_columns)
        
        if_end = False 
        step_number = 0
        sum_of_discounted_rewards = 0
        cumulated_gamma = 1
        path = []

        while (if_end == False):                           # episode steps loop
            step_number += 1
            
            # square region observed by agent:
            observation = afun.observable_region(map, obs_size, position, if_cross)
            #print(str(observation))
            action = my_action(strategy, observation) 

            new_position, reward = afun.transition_and_reward(map,position,action)

            path.append([*position, action, *new_position, reward])

            if (reward > 0)|(step_number > max_num_of_steps):
                if_end = True

            position = new_position
            sum_of_discounted_rewards += reward*cumulated_gamma
            cumulated_gamma *= gamma

        mean_sum_of_discounted_rewards += sum_of_discounted_rewards/number_of_episodes

        print("episode " + str(epi) + ": steps = " + str(step_number) + " sum_of_rewards = " + str(sum_of_discounted_rewards))

        if epi < 5:
            afun.save_map_and_path(map, path, epi)
            afun.save_text_animation(map, path, epi)

    print("after " + str(number_of_episodes) + " episodes:")
    print("mean sum of discounted rewards = " + str(mean_sum_of_discounted_rewards))
    return mean_sum_of_discounted_rewards
    
strategy = animat_train(type_of_map, obs_size, if_cross, log=True)
mean_sum_of_discounted_rewards = animat_test(strategy, type_of_map, obs_size, if_cross)
wandb.log({
    "mean_sum_of_discounted_rewards": mean_sum_of_discounted_rewards,
    })

wandb.finish()
