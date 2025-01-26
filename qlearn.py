import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import board_games_fun as bfun


# Neural network to approximate Q-values
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # Fewer filters
        #     nn.ReLU(),
        #     nn.Flatten()
        # )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(8 * input_size * input_size, 32),  # Smaller hidden layer
            nn.Linear(9, 32),  # Smaller hidden layer
            nn.ReLU(),
            # nn.Linear(32, 16),  # Smaller hidden layer
            # nn.ReLU(),
            nn.Linear(32, output_size)  # Output matches the number of actions
        )

    def forward(self, x):
        # x = self.conv_layers(x)
        return self.fc_layers(x)


# Q-learning Agent
class QLearningAgent:
    def __init__(self, game, learning_rate=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        self.game = game
        self.board_size = game.initial_state().shape[0]
        self.action_size = self.board_size * self.board_size
        self.q_net = QNetwork(self.board_size, self.action_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state, player):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        q_values = self.q_net(state_tensor).detach().numpy().flatten()
        possible_actions = self.game.actions(state, player)
        action_indices = [a[0] * self.board_size + a[1] for a in possible_actions]
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action_index = np.random.choice(action_indices)
        else:
            valid_q_values = {index: q_values[index] for index in action_indices}
            action_index = max(valid_q_values, key=valid_q_values.get)

        action = [action_index // self.board_size, action_index % self.board_size]
        return action

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        action_index = action[0] * self.board_size + action[1]

        self.optimizer.zero_grad()
        q_values = self.q_net(state_tensor)
        target_q_values = q_values.clone()

        # Calculate the target value
        if done:
            target_value = reward
        else:
            next_q_values = self.q_net(next_state_tensor).detach()
            target_value = reward + self.gamma * torch.max(next_q_values)

        # Update the Q-value for the selected action
        target_q_values[0, action_index] = target_value

        # Compute loss and backpropagate
        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()


# Training function
def train_q_learning(game, agent_x, agent_o, num_episodes=2000):
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}")

        state = game.initial_state()
        player = 1
        done = False

        # Inside the train_q_learning function
        move_count = 0
        while not done:
            # Skip action selection if the game is over
            if game.end_of_game(0, move_count, state, [0, 0]):  # Adjust as per your game logic
                break

            agent = agent_x if player == 1 else agent_o
            action = agent.choose_action(state, player)
            next_state, reward = game.next_state_and_reward(player, state, action)

            # Train the agent
            agent.train(state, action, reward, next_state, done)

            state = next_state
            player = 3 - player  # Switch player

            if game.end_of_game(reward, 0, state, action):
                done = True
            move_count += 1

        # Decay epsilon after each episode
        agent_x.update_epsilon()
        agent_o.update_epsilon()


# Testing function
def test_q_learning(game, agent_x, agent_o, num_games=100):
    num_win_x = 0
    num_win_o = 0
    num_draws = 0

    for _ in range(num_games):
        state = game.initial_state()
        player = 1
        done = False

        moves_count=0
        while not done:
            moves_count+=1
            agent = agent_x if player == 1 else agent_o
            action = agent.choose_action(state, player)
            next_state, reward = game.next_state_and_reward(player, state, action)

            state = next_state
            player = 3 - player

            if game.end_of_game(reward, moves_count, state, action):
                done = True
                if reward == 1:
                    num_win_x += 1
                elif reward == -1:
                    num_win_o += 1
                else:
                    num_draws += 1
            

    print(f"X Wins: {num_win_x}, O Wins: {num_win_o}, Draws: {num_draws}")
    return num_win_x, num_win_o, num_draws


def experiment_q_learning_with_testing():
    print("\nQ-Learning Training and Testing\n")
    
    game = bfun.Tictactoe()
    
    # Define Q-learning agents for both players
    agent_x = QLearningAgent(
        game=game,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1,
        epsilon_decay=0.99,
        min_epsilon=0.1
    )
    agent_o = QLearningAgent(
        game=game,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1,
        epsilon_decay=0.99,
        min_epsilon=0.1
    )
    
    # Train the Q-learning agents
    train_q_learning(game, agent_x, agent_o, num_episodes=2000)
    
    # Test trained strategies
    print("Testing trained strategies:")
    num_win_x, num_win_o, num_draws = test_q_learning(game, agent_x, agent_o)
    print(f"Results: X wins: {num_win_x}, O wins: {num_win_o}, Draws: {num_draws}")
    
    # Test partially random strategies
    print("Testing X strategy with partially random O strategy:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    
    for i in range(10):
        epsilon = i / 10
        t.append(epsilon)
        
        # Apply epsilon-greedy randomness to O agent
        agent_o.epsilon = epsilon
        
        wins_x, wins_o, draws = test_q_learning(game, agent_x, agent_o)
        nwin_x.append(wins_x)
        nwin_o.append(wins_o)
        ndraws.append(draws)
    
    # Reset O agent to pure (no randomness)
    agent_o.epsilon = agent_o.min_epsilon
    
    # Plotting results for random O strategy
    plt.plot(t, nwin_x, "x", label="X wins")
    plt.plot(t, nwin_o, "o", label="O wins")
    plt.plot(t, ndraws, "-", label="Draws")
    plt.title("Testing X strategy with partially random O strategy")
    plt.xlabel("Randomness (epsilon) of O strategy")
    plt.ylabel("Number of games")
    plt.legend()
    plt.savefig("test_results_random_o_strategy.png")
    plt.show()
    
    print("Testing O strategy with partially random X strategy:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    
    for i in range(10):
        epsilon = i / 10
        t.append(epsilon)
        
        # Apply epsilon-greedy randomness to X agent
        agent_x.epsilon = epsilon
        
        wins_x, wins_o, draws = test_q_learning(game, agent_x, agent_o)
        nwin_x.append(wins_x)
        nwin_o.append(wins_o)
        ndraws.append(draws)
    
    # Reset X agent to pure (no randomness)
    agent_x.epsilon = agent_x.min_epsilon
    
    # Plotting results for random X strategy
    plt.plot(t, nwin_x, "x", label="X wins")
    plt.plot(t, nwin_o, "o", label="O wins")
    plt.plot(t, ndraws, "-", label="Draws")
    plt.title("Testing O strategy with partially random X strategy")
    plt.xlabel("Randomness (epsilon) of X strategy")
    plt.ylabel("Number of games")
    plt.legend()
    plt.savefig("test_results_random_x_strategy.png")
    plt.show()

# Call the experiment function
experiment_q_learning_with_testing()
