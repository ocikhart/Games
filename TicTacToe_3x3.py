import numpy as np
from itertools import product
import pandas as pd
import tensorflow as tf
import TicTacToe as ttr
import TicTacNN as ttn
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.optimizers import adam_v2 as Adam


def compute_loss(player, experiences, gamma):
    """ 
    Calculates the loss.
    
    Args:
      player (TicTacNN)     :The player to learn by cummulated experiences and q_network and Target_q_network
      experiences (list)    :[[state, reward, done_val] ...]
      gamma (float)         :The discount factor
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    #print(experiences)
    e = list(map(list, zip(*experiences)))
    states = np.asarray(e[0])
    rewards = np.asarray(e[1])
    done_vals = np.asarray(e[2])
    # Compute max Q^(s,a)
    max_qsa = np.full(len(states), -1.)
    for i, s in enumerate(states):
        idx = ttr.TicTacToe.state_index(s)
        if np.size(player.next_states[idx]) == 0:
            continue
        qsa = float(tf.reduce_max(np.ravel(player.target_q_network(player.next_states[idx]))*player.next_states_p[idx], axis=-1))
        if qsa > max_qsa[i]:
            max_qsa[i] = qsa
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa
    
    # Get the q_values
    q_values = np.ravel(player.q_network(states))
    # Compute the loss
    print(y_targets)
    print(q_values)
    loss = MSE(y_targets, q_values) 
    
    return loss

@tf.function
def player_learn(player, experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """ 
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(player, experiences, gamma)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, player.q_network.trainable_variables)
    
    # Update the weights of the q_network.
    player.optimizer.apply_gradients(zip(gradients, player.q_network.trainable_variables))

    # update the weights of target q_network
    player.update_target_network()


def play_games(n, player_X, player_O):
    """
    Simulates N games - no reset of experiences, incrementing only
    
    Args:
      n (int):                  Play N=n games
      player_X (TicTacToe):     Player X
      player_O (TicTacToe):     Player O
    """
    wins_X = 0
    wins_O = 0
    draws = 0
    for j in range(n):
        player_X.reset_board()
        player_O.reset_board()
        i = 1
        moved_O = False
        while True:
            action_X = player_X.pick_best_action('greedy')
            reward_X = player_X.my_move(action_X)
            board_X = np.copy(player_X.board)
            board_O = np.copy(player_O.board)
            reward_O = player_O.opponent_move(action_X)
            if player_X.win(player_X.me):
                print(f"WINNING MOVE of player {player_X.me} ---> Winning Sequence {player_X.winning_sequence}")
                wins_X += 1
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if not np.any(player_X.board == 0):
                print(f"DRAW")
                draws += 1
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if moved_O:
                player_O.experiences.append([np.ravel(board_O), reward_O, 0])
            
            action_O = player_O.pick_best_action('greedy')
            reward_O = player_O.my_move(action_O)
            board_O = np.copy(player_O.board)
            board_X = np.copy(player_X.board)
            reward_X = player_X.opponent_move(action_O)
            moved_O = True
            if player_O.win(player_O.me):
                print(f"WINNING MOVE of player {player_O.me} ---> Winning Sequence {player_O.winning_sequence}")
                wins_O += 1
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            if not np.any(player_O.board == 0):
                print(f"DRAW")
                draws += 1
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            player_X.experiences.append([np.ravel(board_X), reward_X, 0])
            i += 1
    return wins_X, draws, wins_O        

def train_games(n, player_X, player_O):
    """
    Simulates N games and trains the player_X
    
    Args:
      n (int):                  Play N=n games
      player_X (TicTacToe):     Player X
      player_O (TicTacToe):     Player O
    """
    for j in range(n):
        player_X.reset_board()
        player_X.experiences = []
        player_O.reset_board()
        player_O.experiences = []
        i = 1
        first_O = False
        while True:
            action_X = player_X.pick_best_action('greedy')
            reward_X = player_X.my_move(action_X)
            board_X = np.copy(player_X.board)
            board_O = np.copy(player_O.board)
            reward_O = player_O.opponent_move(action_X)
            if player_X.win(player_X.me):
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if not np.any(player_X.board == 0):
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if first_O:
                player_O.experiences.append([np.ravel(board_O), reward_O, 0])
            
            action_O = player_O.pick_best_action('greedy')
            reward_O = player_O.my_move(action_O)
            board_O = np.copy(player_O.board)
            board_X = np.copy(player_X.board)
            reward_X = player_X.opponent_move(action_O)
            first_O = True
            if player_O.win(player_O.me):
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            if not np.any(player_O.board == 0):
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            player_X.experiences.append([np.ravel(board_X), reward_X, 0])
            i += 1



INPUT = 9
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.

tf.random.set_seed(SEED)

t_board_X = ttn.TicTacNN(player = 1,reward_type ='goal_reward')
t_board_O = ttr.TicTacToe(player = 2,reward_type ='goal_reward')

#state = np.array([1,2,1,0,2,1,0,0,0])
#print(t_board_X.next_states[ttr.TicTacToe.state_index(state)])
#print(t_board_X.next_states_p[ttr.TicTacToe.state_index(state)])
#print(len(t_board_X.next_states[ttr.TicTacToe.state_index(state)]))
wins_X, draws, wins_O = play_games(4, t_board_X, t_board_O)
print(f"{wins_X} : {draws} : {wins_O}")
loss = compute_loss(t_board_X, t_board_X.experiences, ttn.GAMMA)
print(loss)