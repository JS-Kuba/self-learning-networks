import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  
        self.current_player = 'X'

    def is_winner(self, player):
        win_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        for positions in win_positions:
            if all(self.board[pos] == player for pos in positions):
                return True
        return False

    def is_full(self):
        return ' ' not in self.board

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def play(self, position):
        self.board[position] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def get_state(self):
        return np.array([1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in self.board])

    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board[:]
        new_game.current_player = self.current_player
        return new_game

# Neural Q-value approximator
class NeuralQApproximator(nn.Module):
    def __init__(self):
        super(NeuralQApproximator, self).__init__()
        self.fc1 = nn.Linear(9, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(0) 


# Q-learning Agent with Neural Network
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha  
        self.gamma = gamma
        self.epsilon = epsilon

        # Neural network model
        self.model = NeuralQApproximator()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def get_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model(state_tensor)

    def choose_action(self, game):
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            return random.choice(game.available_moves())  # Explore
        else:
            # Exploit: choose the action with the maximum Q-value
            q_values = self.get_q_values(state_tensor)

            # Get Q-values for the available actions only
            available_moves = game.available_moves()

            # Ensure that the q_values tensor has the correct shape for indexing
            available_q_values = q_values[0, available_moves].detach().numpy()

            # Choose the action with the max Q-value among the available moves
            best_move = available_moves[np.argmax(available_q_values)]  # Select the move with the highest Q-value
            return best_move




    def update(self, state, action, reward, next_state, next_available_moves):
        # Get Q-values for current state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        current_q_values = self.get_q_values(state_tensor)

        # Get max Q-value for next state
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        next_q_values = self.get_q_values(next_state_tensor)
        max_future_q = max(next_q_values[0][next_available_moves].detach().numpy(), default=0)

        # Update Q-values using the Q-learning update rule
        target = current_q_values[0][action].item() + self.alpha * (reward + self.gamma * max_future_q - current_q_values[0][action].item())
        
        # Compute loss and backpropagate
        loss = self.loss_fn(current_q_values[0][action], torch.tensor(target, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


# Training function
def train_q_agents(num_games=1000):
    agent_x = QLearningAgent()
    agent_o = QLearningAgent()

    for _ in tqdm(range(num_games), "Training Q-learning agents"):
        game = TicTacToe()
        states_actions_x = []
        states_actions_o = []

        while not game.is_full() and not game.is_winner('X') and not game.is_winner('O'):
            current_player = game.current_player
            state = game.get_state()
            if current_player == 'X':
                action = agent_x.choose_action(game)
                states_actions_x.append((state, action))
            else:
                action = agent_o.choose_action(game)
                states_actions_o.append((state, action))
            game.play(action)

        if game.is_winner('X'):
            reward_x, reward_o = 1, -1
        elif game.is_winner('O'):
            reward_x, reward_o = -1, 1
        else:
            reward_x, reward_o = 0.5, 0.5

        for state, action in states_actions_x:
            next_state = game.get_state()
            agent_x.update(state, action, reward_x, next_state, game.available_moves())

        for state, action in states_actions_o:
            next_state = game.get_state()
            agent_o.update(state, action, reward_o, next_state, game.available_moves())

    return agent_x, agent_o


def test_q_agents(agent_x, agent_o, num_games=100, epsilon_values=None):
    if epsilon_values is None:
        epsilon_values = [0.0]

    results = []
    for epsilon in epsilon_values:
        agent_x.set_epsilon(epsilon)

        wins_x, wins_o, draws = 0, 0, 0

        for _ in range(num_games):
            game = TicTacToe()
            while not game.is_full() and not game.is_winner('X') and not game.is_winner('O'):
                if game.current_player == 'X':
                    move = agent_x.choose_action(game)
                else:
                    move = agent_o.choose_action(game)
                game.play(move)

            if game.is_winner('X'):
                wins_x += 1
            elif game.is_winner('O'):
                wins_o += 1
            else:
                draws += 1

        results.append((epsilon, wins_x, wins_o, draws))
    return results


# Main execution
if __name__ == "__main__":
    num_training_games = 1000
    num_test_games = 100

    # Train agents
    agent_x, agent_o = train_q_agents(num_training_games)

    # Test agents with varying randomness for O
    epsilon_values = [i / 10 for i in range(11)]  # Epsilon from 0.0 to 1.0
    results = test_q_agents(agent_x, agent_o, num_test_games, epsilon_values)

    # Display results
    for epsilon, wins_x, wins_o, draws in results:
        print(f"Epsilon: {epsilon:.1f}, X wins: {wins_x}, O wins: {wins_o}, Draws: {draws}")


    epsilons, x_wins, o_wins, draws = zip(*results)
    plt.plot(epsilons, x_wins, "x-", label="X wins")
    plt.plot(epsilons, o_wins, "o-", label="O wins")
    plt.plot(epsilons, draws, "s-", label="Draws")
    plt.xlabel(f"Randomness of X's strategy (epsilon)")
    plt.ylabel("Number of games")
    plt.legend()
    plt.title(f"Neural Q-learning: O strategy vs X with varying randomness")
    plt.show()
