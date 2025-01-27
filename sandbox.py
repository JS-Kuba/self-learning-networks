import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# TicTacToe class remains the same
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # 3x3 board flattened into a list
        self.current_player = 'X'  # 'X' always starts

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
        return ''.join(self.board)

    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board[:]
        new_game.current_player = self.current_player
        return new_game

# Node and MCTS classes
class Node:
    def __init__(self, game, parent=None):
        self.game = game
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = game.available_moves()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4):
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            ucb_value = (
                child.wins / (child.visits + 1e-6) +
                exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            )
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        return best_child

class MCTS:
    def __init__(self, game, max_simulations=1000):
        self.game = game
        self.max_simulations = max_simulations
        self.root = Node(game)

    def tree_policy(self, node):
        while not node.game.is_full() and not node.game.is_winner('X') and not node.game.is_winner('O'):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        action = random.choice(node.untried_actions)
        new_game = node.game.copy()
        new_game.play(action)
        child_node = Node(new_game, parent=node)
        node.children.append(child_node)
        node.untried_actions.remove(action)
        return child_node

    def simulate(self, node):
        current_game = node.game.copy()
        while not current_game.is_full() and not current_game.is_winner('X') and not current_game.is_winner('O'):
            move = random.choice(current_game.available_moves())
            current_game.play(move)
        if current_game.is_winner('X'):
            return 1  # X wins
        elif current_game.is_winner('O'):
            return 0  # O wins
        else:
            return 0.5  # Draw

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    # def best_move(self, epsilon=0.0):
    #     if random.random() < epsilon:
    #         return random.choice(self.game.available_moves())
    #     for _ in range(self.max_simulations):
    #         node = self.tree_policy(self.root)
    #         result = self.simulate(node)
    #         self.backpropagate(node, result)

    #     best_child = self.root.best_child(exploration_weight=0)
    #     if best_child is not None:
    #         # Get the move leading to the best child
    #         for action in self.root.untried_actions:
    #             new_game = self.game.copy()
    #             new_game.play(action)
    #             if new_game.get_state() == best_child.game.get_state():
    #                 return action
    #     return random.choice(self.game.available_moves())  # Fallback in case of an issue

# QLearning class
class QLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # współczynnik uczenia
        self.gamma = gamma  # współczynnik dyskontowania
        self.epsilon = epsilon  # parametr epsilon dla eksploracji
        self.q_table = {}  # tabela Q - przechowuje Q(s, a)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q(self, state, action, reward, next_state, next_action):
        max_q_next = max([self.get_q(next_state, a) for a in TicTacToe().available_moves()])
        new_q = (1 - self.alpha) * self.get_q(state, action) + self.alpha * (reward + self.gamma * max_q_next)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state, available_actions, eps=0.1):
        if random.random() < eps:
            # Eksploracja: wybór losowej akcji
            return random.choice(available_actions)
        else:
            # Eksploatacja: wybór akcji z najlepszą wartością Q
            return max(available_actions, key=lambda a: self.get_q(state, a))


class MCTSWithQLearning(MCTS):
    def __init__(self, game, q_learning, max_simulations=1000):
        super().__init__(game, max_simulations)
        self.q_learning = q_learning

    def best_move(self, epsilon=0.1):
        # Get current state
        state = self.game.get_state()
        available_actions = self.game.available_moves()

        # Epsilon-Greedy action selection using Q-learning
        action = self.q_learning.choose_action(state, available_actions, epsilon)

        # Perform MCTS simulations (No need to pass epsilon to simulate method)
        for _ in range(self.max_simulations):
            node = self.tree_policy(self.root)
            result = self.simulate(node)  # Simulate without epsilon
            self.backpropagate(node, result)

        # After MCTS simulations, update the Q-value table based on the result
        best_child = self.root.best_child(exploration_weight=0)
        if best_child is not None:
            next_state = best_child.game.get_state()
            reward = result  # Result from MCTS simulation
            self.q_learning.update_q(state, action, reward, next_state, action)

        return action

def train_agents_with_qlearning(num_games=1000):
    q_learning_x = QLearning()
    q_learning_o = QLearning()

    agent_x = MCTSWithQLearning(TicTacToe(), q_learning_x, max_simulations=10)
    agent_o = MCTSWithQLearning(TicTacToe(), q_learning_o, max_simulations=10)

    for _ in tqdm(range(num_games), "Train"):
        game = TicTacToe()
        agent_x.root = Node(game)
        agent_o.root = Node(game)

        while not game.is_full() and not game.is_winner('X') and not game.is_winner('O'):
            if game.current_player == 'X':
                move = agent_x.best_move()
            else:
                move = agent_o.best_move()
            game.play(move)

    return agent_x, agent_o

def test_agents_with_qlearning(agent_x, agent_o, num_games=100, epsilon_x=0.1, epsilon_o=0.1):
    wins_x, wins_o, draws = 0, 0, 0

    for _ in tqdm(range(num_games), "Test"):
        game = TicTacToe()
        agent_x.root = Node(game)
        agent_o.root = Node(game)

        while not game.is_full() and not game.is_winner('X') and not game.is_winner('O'):
            if game.current_player == 'X':
                move = agent_x.best_move(epsilon=epsilon_x)  # Use epsilon_x for agent_x
            else:
                move = agent_o.best_move(epsilon=epsilon_o)  # Use epsilon_o for agent_o
            game.play(move)

        if game.is_winner('X'):
            wins_x += 1
        elif game.is_winner('O'):
            wins_o += 1
        else:
            draws += 1

    return wins_x, wins_o, draws


if __name__ == "__main__":
    trained_agent_x, trained_agent_o = train_agents_with_qlearning(num_games=1000)

    epsilons = [i / 10 for i in range(11)]
    results = {"wins_x": [], "wins_o": [], "draws": []}

    for epsilon_o in epsilons:  
        epsilon_x = 0.1
        wins_x, wins_o, draws = test_agents_with_qlearning(
            trained_agent_x, trained_agent_o, num_games=50, epsilon_x=epsilon_x, epsilon_o=epsilon_o
        )
        results["wins_x"].append(wins_x)
        results["wins_o"].append(wins_o)
        results["draws"].append(draws)

    # Plotting results
    plt.plot(epsilons, results["wins_x"], "x-", label="X wins")
    plt.plot(epsilons, results["wins_o"], "o-", label="O wins")
    plt.plot(epsilons, results["draws"], "^-", label="Draws")
    plt.xlabel("Randomness (epsilon) of Player O")
    plt.ylabel("Number of Games")
    plt.title("Performance of X vs Partially Random O")
    plt.legend()
    plt.show()

