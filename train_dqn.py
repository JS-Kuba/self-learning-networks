import random

import numpy as np
import torch

from tqdm import tqdm

import parking_model as pm
from model import DQN

GLOBAL_VARS = pm.GlobalVar()

ALPHA = 0.001 
EPS_START = 0.9
EPS_END = 0.05
GAMMA = 0.99

number_of_episodes = 2000

## Actions
PREDEFINED_ACTIONS = [
    [angle, speed]
    for angle in np.arange(
        -GLOBAL_VARS.wheel_turn_angle_max,
        GLOBAL_VARS.wheel_turn_angle_max + np.pi / 8,
        np.pi / 8,
    )
    for speed in np.arange(-GLOBAL_VARS.Vmod, GLOBAL_VARS.Vmod + 1, 1)
    if not speed == 0
]
# PREDEFINED_ACTIONS.append([0, 0])

## Batching
BATCH_SIZE = 10
experience_buffer = []

## Logging
episode_rewards = []
episode_steps = []

### Code for neural network training
DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMBER_OF_FEATURES = 3 # 3 for state (x, y, angle)
NUMBER_OF_ACTIONS = len(PREDEFINED_ACTIONS)

def optimize_model():
    if len(experience_buffer) < BATCH_SIZE:
        return

    batch = random.sample(experience_buffer, BATCH_SIZE)

    state_batch = torch.cat([state for state, _, _, _ in batch])
    action_batch = torch.cat([action for _, action, _, _ in batch])
    reward_batch = torch.cat([reward for _, _, reward, _ in batch])
    next_state_batch = torch.cat([next_state for _, _, _, next_state in batch])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Q-values for the next states using the policy_net
    with torch.inference_mode():
        next_state_values = target_net(next_state_batch).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


## Neural Model
LR = 1e-4
TAU = 0.005
policy_net = DQN(NUMBER_OF_FEATURES, NUMBER_OF_ACTIONS).to(DEVICE)
target_net = DQN(NUMBER_OF_FEATURES, NUMBER_OF_ACTIONS).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
##############################

def reward_function(state, if_collision, if_stopped):
    distance_to_goal = np.sqrt(state[0] ** 2 + state[1] ** 2)
    angle_error = min(abs(state[2]), abs(abs(state[2]) - np.pi))

    distance_reward = -np.clip(distance_to_goal, 0, 1)
    angle_reward = np.cos(angle_error)

    if if_collision:
        return -1.0
    elif if_stopped:
        return distance_reward + angle_reward
    else:
        return 0.1 * distance_reward


def check_if_stopped(state) -> bool:
    x, y, angle = state
    distance_to_goal = np.sqrt(x**2 + y**2)
    angle_error = min(abs(angle), abs(abs(angle) - np.pi))

    goal_tolerance = 1
    angle_tolerance = np.radians(5)

    return distance_to_goal < goal_tolerance and angle_error < angle_tolerance


def choose_action(state):
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.inference_mode():
        action_idx = policy_net(state).max(1).indices.view(1, 1)

    action = PREDEFINED_ACTIONS[action_idx]
    angle, V = action
    if_stopped = check_if_stopped(state.squeeze()) or V == 0
    return angle, V, if_stopped

import os
def get_best():
    if os.path.exists("best_run.txt"):
        with open("best_run.txt", 'r') as best_run_file:
            try:
                max_ocena = float(best_run_file.readline())
            except ValueError:
                max_ocena = 0
    else:
        max_ocena = 0
    return max_ocena

# test parkowania - nie wolno niczego zmieniać!
def park_test(param_fiz, stany_poczatkowe, model, nazwa_pliku):
    max_ocena = get_best()
    pm.park_save("param.txt", param_fiz)
    phist = open(nazwa_pliku, "w")
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape
    sr_ocena_koncowa = 0
    sr_liczba_krokow = 0
    for epizod in range(liczba_stanow_poczatkowych):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod % liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup, :]
        run_history = [] 
        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while czy_zatrzymanie == False:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan zgodnie z wyuczoną strategią:
            kat, V, czy_zatrzymanie = choose_action(stan)

            # zapis kroku historii:
            # phist.write(str(epizod + 1) + "  " + str(krok) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(kat) + "  " + str(V) + "\n")
            phist.write(
                "%d %d %.4f %.4f %.4f %.4f %.4f\n"
                % ((epizod + 1), krok, stan[0], stan[1], stan[2], kat, V)
            )
            run_history.append((epizod + 1, krok, stan[0], stan[1], stan[2], kat, V))

            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, stan, kat, V)

            if (czy_kolizja) | (krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            stan = nowystan
        ocena_koncowa = pm.final_score(param_fiz, nowystan, czy_kolizja, krok)

        if ocena_koncowa > max_ocena and krok > 10:
            max_ocena = ocena_koncowa
            with open("best_run.txt", 'w') as best_run_file:
                best_run_file.write(f"{ocena_koncowa}\n")
                for record in run_history:
                    best_run_file.write("%d %d %.4f %.4f %.4f %.4f %.4f\n" % record)

        sr_ocena_koncowa += ocena_koncowa / liczba_stanow_poczatkowych
        sr_liczba_krokow = sr_liczba_krokow + krok / liczba_stanow_poczatkowych
        print(
            "w %d epizodzie ocena parkowania = %g, liczba krokow = %d"
            % (epizod, ocena_koncowa, krok)
        )

    print("srednia ocena końcowa na epizod = %g" % (sr_ocena_koncowa))
    print("srednia liczba krokow = %g" % (sr_liczba_krokow))
    phist.close()
    return sr_ocena_koncowa


# Wybór akcji z polityką epsilon-greedy
def epsilon_greedy_policy(state, epsilon, step=0):
    if_stopped = False
    if np.random.rand() < epsilon:
        action_idx = torch.tensor(np.random.choice(len(PREDEFINED_ACTIONS)), device=DEVICE).view(1, 1)
    else:
        with torch.inference_mode():
            action_idx = policy_net(state).max(1).indices.view(1, 1)
    if step > 200:
        if_stopped = True
    return action_idx, if_stopped

def park_train():
    epsilon = EPS_START

    stany_poczatkowe_1 = np.array(
        [
            [9.1, 4.6, 0],
            [6.3, 5.06, 0],
            [9.6, 3.15, 0],
            [7.3, 5.75, 0],
            [10.1, 6.21, 0],
        ],
        dtype=float,
    )
    stany_poczatkowe = stany_poczatkowe_1
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape

    weights = 0


    for episode in tqdm(range(number_of_episodes)):
        epsilon = max(EPS_END, epsilon * 0.99)

        nr_stanup = episode % liczba_stanow_poczatkowych
        state = stany_poczatkowe[nr_stanup, :]
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        step = 0
        if_collision = False
        if_stopped = False
        total_reward = 0
        while not if_stopped:
            step = step + 1

            action_idx, if_stopped = epsilon_greedy_policy(state, epsilon, step)
            angle, V = PREDEFINED_ACTIONS[action_idx]

            next_state, rotation_center, if_collision = pm.model_of_car(
                GLOBAL_VARS, state.squeeze(), angle, V
            )
            next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            if if_collision or (step >= GLOBAL_VARS.max_number_of_steps):
                if_stopped = True

            reward = reward_function(next_state.squeeze(), if_collision, if_stopped)

            reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)

            experience_buffer.append((state, action_idx, reward, next_state))
            optimize_model()

            state = next_state
            total_reward += reward

            # Soft update of target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        episode_rewards.append(total_reward)
        episode_steps.append(step)

        if episode % 100 == 0:
            print("epizod %d\n" % episode)
            park_test(GLOBAL_VARS, stany_poczatkowe, weights, "historia_park.txt")

    # sprawdzenie czy system dobrze uogólnia dla dowolnych stanów początkowych:
    stany_pocz_losowe = pm.random_initial_states(pm.GlobalVar(), 20)
    print("Test dla losowych stanów początkowych:")
    park_test(GLOBAL_VARS, stany_pocz_losowe, weights, "historia_park_los.txt")


ocena_koncowa_maks = pm.final_score(
    pm.GlobalVar(), [0, 0, -np.pi], if_collision=False, num_of_steps=100
)
print("najlepsza możliwa ocena końcowa = " + str(ocena_koncowa_maks))

park_train()