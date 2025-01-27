import numpy as np
import pickle
import time
import matplotlib.pylab as plt
import pdb
import board_games_fun as bfun
import board_graphical_interface as bgra
import tensorflow as tf
from tensorflow.keras import layers

# Q-learning xo strategy using dictionary for Q-values in each known state: 

class Strategy_PolicyNeuro:
    def __init__(self, game_obj):
        self.game_obj = game_obj
        self.model = self.build_model()

    def build_model(self):
        # Build a simple neural network with one hidden layer
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.game_obj.state_size(),)),  # Define input shape
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.game_obj.num_actions(), activation='linear')  # Output layer (Q-values for each action)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def choose_action(self, state, player):
        Q_values = self.predict(state)
        action_no = np.argmax(Q_values)
        return action_no, Q_values[action_no]

    def predict(self, state):
        state_input = np.array([state])
        return self.model.predict(state_input)[0]

    def train(self, state, action, target):
        with tf.GradientTape() as tape:
            Q_values = self.model(state)
            loss = tf.keras.losses.MSE(Q_values[action], target)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class Strategy_VNeuro:
    def __init__(self, game_obj):
        self.game_obj = game_obj
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.game_obj.state_size(),)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)  # Output layer (single value for state value)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, state):
        state_input = np.array([state])
        return self.model.predict(state_input)[0]

    def train(self, state, target):
        with tf.GradientTape() as tape:
            value = self.model(state)
            loss = tf.keras.losses.MSE(value, target)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

class Node:
    def __init__(self, state, action=None):
        self.state = state
        self.action = action
        self.parent = None
        self.children = []
        self.reward = 0
        self.visits = 0

    def fully_expanded(self):
        return len(self.children) == len(self.game_obj.actions(self.state))


class MCTS:
    def __init__(self, game_obj, strategy):
        self.game_obj = game_obj
        self.strategy = strategy
        self.max_simulations = 100  # Max number of simulations per step

    def search(self, state, player):
        root = Node(state)
        for _ in range(self.max_simulations):
            node = self.tree_policy(root, player)
            reward = self.default_policy(node.state, player)
            self.backpropagate(node, reward)
        return self.best_action(root)

    def tree_policy(self, node, player):
        while not self.game_obj.end_of_game(node.state):
            if not node.fully_expanded():
                return self.expand(node, player)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node, player):
        untried_actions = self.game_obj.actions(node.state, player)
        action = untried_actions[0]  # Pick the first available action
        next_state = self.game_obj.next_state(player, node.state, action)
        child_node = Node(next_state, action)
        node.children.append(child_node)
        return child_node

    def best_child(self, node):
        # Use Q-values from the neural network for best child selection
        values = [self.strategy.predict(child.state) for child in node.children]
        best_value_idx = np.argmax(values)
        return node.children[best_value_idx]

    def default_policy(self, state, player):
        # Default policy for simulation (random game play or using trained strategy)
        while not self.game_obj.end_of_game(state):
            actions = self.game_obj.actions(state, player)
            action = np.random.choice(actions)  # Random action in simulation
            state = self.game_obj.next_state(player, state, action)
            player = 3 - player  # Switch player
        return self.game_obj.reward(state)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def best_action(self, root):
        # Return the best action based on the visit count or reward
        return max(root.children, key=lambda x: x.visits).action

def board_game_train_Q2_with_MCTS(game_object, players_to_train, strategy_x=None, strategy_o=None, number_of_games=2000):
    mcts_x = MCTS(game_object, strategy_x)
    mcts_o = MCTS(game_object, strategy_o)

    for game_nr in range(number_of_games):
        State = game_object.initial_state()
        player = 1
        while not game_object.end_of_game(State):
            if player == 1:
                action = mcts_x.search(State, player)
            else:
                action = mcts_o.search(State, player)

            State = game_object.next_state(player, State, action)
            player = 3 - player  # Switch player
    return strategy_x, strategy_o


# Test of given game (game_object) and strategies for player x and o
# choose_random - numbers of moves with puting x or o into random empty cell 
# (odd numbers {1,3,5,7,9} for x, even numbers {2,4,6,8} for o):
def board_game_test(game_object, strategy_x, strategy_o, number_of_games = 100, choose_random = []):

    num_win_x = 0
    num_win_o = 0
    num_draws = 0
    
    Games = []
    Rewards = []

    for game in range(number_of_games):                   # episodes loop
        #print("game = " + str(game))
        State = game_object.initial_state()               # initial state - empty board in tictac
        player = 1                                        # first movement by player 1(cross)
        if_end = False 
        step_number = 0
        States = []
        Actions = []

        States.append(State)

        while (if_end == False):                           # episode steps loop
            step_number += 1

            actions = game_object.actions(State, player)   

            if player == 1:
                strategy = strategy_x
            else:
                strategy = strategy_o
            
            action_nr , value = strategy.choose_action(State,player)
            if (action_nr == None) | (step_number in choose_random):
                action_nr = np.random.randint(len(actions))
      
            NextState, Reward =  game_object.next_state_and_reward(player, State, actions[action_nr])

            State = NextState                                        # move to next state
            Actions.append(actions[action_nr])
            States.append(State)                                     # board for game description

            player = 3 - player                                      # player changing

            if game_object.end_of_game(Reward, step_number,State,action_nr):      # win or draw
                if_end = True
                if Reward == 1:
                    num_win_x += 1
                elif Reward == -1:
                    num_win_o += 1
                elif Reward == 0:
                    num_draws += 1
                Rewards.append(Reward)
        Games.append([States, Actions])
    return num_win_x, num_win_o, num_draws, Games, Rewards



def experiment_par_train():
    print("\nUCZENIE DWOCH STRATEGII JEDNOCZESNIE\n")
    game = bfun.Tictactoe()                      # game class object

    strategy_x, strategy_o = board_game_train_Q2_with_MCTS(game,players_to_train = [1,2], number_of_games = 2000)

    print("test stategii uczonych jednocześniewl,:")
    num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game,strategy_x,strategy_o,choose_random=[])
    print("liczby wygranych: x = "+str(num_win_x)+", o = "+str(num_win_o) + ", l.remisów = "+str(num_draws))
    game.print_test_to_file("gry_wyuczonych_strategii.txt",num_win_x, num_win_o, num_draws, Games, Rewards)
    
    print("test stategii x na częściowo losowej o:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        epsilon = i/10
        t.append(epsilon)     

        strategy_o.make_epsilon_greedy(player=2,epsilon=epsilon)
        num_win_x, num_win_o, num_draws, Games, Rewards =\
              board_game_test(game, strategy_x, strategy_o)
        game.print_test_to_file("gry_x_vs_losowe_o_epsilon"+str(epsilon)+".txt",\
                                num_win_x, num_win_o, num_draws, Games, Rewards)
        
        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    strategy_o.make_pure()
    plt.plot(t,nwin_x,"x",t,nwin_o,"o",t, ndraws,"-")
    plt.title("tictac test results with Nash x strategy and partially random o strategy")
    plt.xlabel("randomness of o strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins","num.of o wins","num of draws"])
    plt.savefig("test_Nash_x_strategy_random_o_strategy.png")
    plt.show()

experiment_par_train()