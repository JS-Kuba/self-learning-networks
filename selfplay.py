import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import board_games_fun as bfun
import matplotlib.pyplot as plt

class StrategyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(StrategyNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * input_size * input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

class Strategy_PolicyNeuro:
    def __init__(self, game_obj, learning_rate=0.001):
        self.game_obj = game_obj
        self.board_size = game_obj.initial_state().shape[0]
        self.action_size = self.board_size * self.board_size
        self.policy_net = StrategyNN(self.board_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.if_epsilon_greedy = False
        self.epsilon=0

    def choose_action(self, state, player):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = self.policy_net(state_tensor)
        action_probs = torch.softmax(logits, dim=1).detach().numpy().flatten()

        possible_actions = self.game_obj.actions(state, player)
        action_indices = [a[0] * self.board_size + a[1] for a in possible_actions]
        valid_probs = action_probs[action_indices]

        if valid_probs.sum() == 0:
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
        else:
            valid_probs /= valid_probs.sum()  # Normalize probabilities
        
        if self.if_epsilon_greedy:
            if np.random.rand() < self.epsilon:
                action_flat = np.random.choice(action_indices)  # Explore
            else:
                action_flat = action_indices[np.argmax(valid_probs)]  # Exploit
        else:
            action_flat = action_indices[np.argmax(valid_probs)]

        action = [action_flat // self.board_size, action_flat % self.board_size]

        return action, valid_probs[action_indices.index(action_flat)]

    def make_epsilon_greedy(self, epsilon):
        self.epsilon = epsilon        
        self.if_epsilon_greedy = True
        
    def make_pure(self):
        self.if_epsilon_greedy = False 
        self.epsilon = 0  


    def train_on_sample(self, state, action, reward):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action_index = action[0] * self.board_size + action[1]
        action_tensor = torch.tensor([action_index], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        self.optimizer.zero_grad()
        logits = self.policy_net(state_tensor)
        loss = self.criterion(logits, action_tensor) * reward_tensor
        loss.backward()
        self.optimizer.step()

class Strategy_VNeuro:
    def __init__(self, game_obj, learning_rate=0.001):
        self.game_obj = game_obj
        self.board_size = game_obj.initial_state().shape[0]
        self.value_net = StrategyNN(self.board_size, 1)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def evaluate_state(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        value = self.value_net(state_tensor).item()
        return value

    def train_on_sample(self, state, target_value):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.tensor([target_value], dtype=torch.float32)

        self.optimizer.zero_grad()
        value = self.value_net(state_tensor)
        loss = self.criterion(value, target_tensor)
        loss.backward()
        self.optimizer.step()

def board_game_train_actor_critic(game_object, strategy_policy, strategy_value, number_of_games=2000):
    gamma = 0.9
    t1 = time.time()

    for game_nr in range(number_of_games):
        if (game_nr * 100) % number_of_games == 0:
            print(f"Game {game_nr}/{number_of_games}")

        state = game_object.initial_state()
        player = 1
        done = False
        trajectory = []

        while not done:
            action, _ = strategy_policy.choose_action(state, player)
            next_state, reward = game_object.next_state_and_reward(player, state, action)

            trajectory.append((state, action, reward))

            state = next_state
            player = 3 - player
            if game_object.end_of_game(reward, len(trajectory), state, action):
                done = True

        G = 0
        for state, action, reward in reversed(trajectory):
            G = reward + gamma * G
            strategy_value.train_on_sample(state, G)
            advantage = G - strategy_value.evaluate_state(state)
            strategy_policy.train_on_sample(state, action, advantage)

    dt = time.time() - t1
    print(f"Training finished in {dt:.2f} seconds.")


def board_game_test(game_object, strategy_x, strategy_o, number_of_games=100, choose_random=[]):
    num_win_x = 0
    num_win_o = 0
    num_draws = 0
    
    Games = []
    Rewards = []

    for game in range(number_of_games):
        state = game_object.initial_state()
        player = 1
        done = False
        step_number = 0
        states = []
        actions = []

        states.append(state)

        while not done:
            step_number += 1

            actions_list = game_object.actions(state, player)

            strategy = strategy_x if player == 1 else strategy_o
            action, _ = strategy.choose_action(state, player)  
            if (action is None) or (step_number in choose_random):
                action = actions_list[np.random.randint(len(actions_list))] 
            elif action not in actions_list:
                raise ValueError(f"Invalid action {action} not in actions list {actions_list}")

            next_state, reward = game_object.next_state_and_reward(player, state, action)

            state = next_state
            actions.append(action)
            states.append(state)

            player = 3 - player  # Switch player

            if game_object.end_of_game(reward, step_number, state, action):
                done = True
                if reward == 1:
                    num_win_x += 1
                elif reward == -1:
                    num_win_o += 1
                elif reward == 0:
                    num_draws += 1
                Rewards.append(reward)
        Games.append([states, actions])
    return num_win_x, num_win_o, num_draws, Games, Rewards


def experiment_actor_critic_with_testing():
    print("\nActor-Critic Training and Testing\n")
    # game = bfun.Tictactoe()
    game = bfun.Tictac_general(4,4,3,True)

    strategy_policy_x = Strategy_PolicyNeuro(game)
    strategy_policy_o = Strategy_PolicyNeuro(game)
    strategy_value = Strategy_VNeuro(game)

    board_game_train_actor_critic(game, strategy_policy_x, strategy_value, number_of_games=1000)

    print("Testing trained strategies:")
    num_win_x, num_win_o, num_draws, _, _ = board_game_test(game, strategy_policy_x, strategy_policy_o)
    print(f"Results: X wins: {num_win_x}, O wins: {num_win_o}, Draws: {num_draws}")

    print("Testing X strategy with partially random O strategy:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []

    for i in range(10):
        epsilon = i / 10
        t.append(epsilon)
        
        strategy_policy_o.make_epsilon_greedy(epsilon=epsilon)
        
        wins_x, wins_o, draws, _, _ = board_game_test(game, strategy_policy_x, strategy_policy_o)
        nwin_x.append(wins_x)
        nwin_o.append(wins_o)
        ndraws.append(draws)

    strategy_policy_o.make_pure()

    plt.plot(t, nwin_x, "x", label="X wins")
    plt.plot(t, nwin_o, "o", label="O wins")
    plt.plot(t, ndraws, "-", label="Draws")
    plt.title("Testing X strategy with partially random O strategy")
    plt.xlabel("Randomness (epsilon) of O strategy")
    plt.ylabel("Number of games")
    plt.legend()
    plt.savefig("test_results_random_o_strategy.png")
    plt.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("Testing O strategy with partially random X strategy:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []

    for i in range(10):
        epsilon = i / 10
        t.append(epsilon)
        
        strategy_policy_x.make_epsilon_greedy(epsilon=epsilon)
        
        wins_x, wins_o, draws, _, _ = board_game_test(game, strategy_policy_x, strategy_policy_o)
        nwin_x.append(wins_x)
        nwin_o.append(wins_o)
        ndraws.append(draws)

    strategy_policy_x.make_pure()

    plt.plot(t, nwin_x, "x", label="X wins")
    plt.plot(t, nwin_o, "o", label="O wins")
    plt.plot(t, ndraws, "-", label="Draws")
    plt.title("Testing O strategy with partially random X strategy")
    plt.xlabel("Randomness (epsilon) of X strategy")
    plt.ylabel("Number of games")
    plt.legend()
    plt.savefig("test_results_random_o_strategy.png")
    plt.show()

experiment_actor_critic_with_testing()
