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


def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa
    
    # Get the q_values
    q_values = q_network(states)
        
    # Compute the loss
    loss = MSE(y_targets, q_values) 
    
    return loss

@tf.function
def agent_learn(agent, experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """ 
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, agent.q_network, agent.target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, agent.q_network.trainable_variables)
    
    # Update the weights of the q_network.
    agent.optimizer.apply_gradients(zip(gradients, agent.q_network.trainable_variables))

    # update the weights of target q_network
    agent.update_target_network()


def play_games(n, player_X, player_O):
    """
    Simulates N games
    
    Args:
      n (int):                  Play N=n games
      player_X (TicTacToe):     Player X
      player_O (TicTacToe):     Player O
    """
    for j in range(n):
        game_over = False
        player_X.reset_board()
        player_O.reset_board()
        i = 1
        while True:
            action_X = t_board_X.pick_best_action('greedy')
            print(f"Match #{j+1} Round #{i} PLAYER X: {action_X}")
            res_X = t_board_X.my_move(action_X)
            print(t_board_X.show_board())
            if not t_board_X.win(t_board_X.me):
                t_board_O.opponent_move(action_X)
            else:
                print(f"WINNING MOVE ---> Winning Sequence {t_board_X.winning_sequence}")
                break
            if not np.any(t_board_X.board == 0):
                print(f"DRAW")
                break
            action_O = t_board_O.pick_best_action('greedy')
            print(f"Match #{j+1} Round #{i} PLAYER O: {action_O}")
            res_X = t_board_O.my_move(action_O)
            print(t_board_O.show_board())
            if not t_board_O.win(t_board_O.me):
                t_board_X.opponent_move(action_O)
            else:
                print(f"WINNING MOVE ---> Winning Sequence {t_board_O.winning_sequence}")
                break
            if not np.any(t_board_O.board == 0):
                print(f"DRAW")
                break
            i += 1

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
        player_O.reset_board()
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
            if not np.any(t_board_O.board == 0):
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            player_X.experiences.append([np.ravel(board_X), reward_X, 0])
            i += 1
        print(player_X.experiences)
        print(player_X.board)


INPUT = 9
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.

tf.random.set_seed(SEED)

t_board_X = ttr.TicTacToe(player = 1,reward_type ='goal_reward')
t_board_O = ttr.TicTacToe(player = 2,reward_type ='goal_reward')

### play_games(2, t_board_X, t_board_O)
### actions =  t_board_X.possible_actions(t_board_X.board)
### actions =  np.array([0, 2, 4, 8])
### states = np.tile(np.ravel(t_board_X.board), (len(actions),1))
### for i,action in enumerate(actions):
###     states[i][action] = t_board_X.me
### print(states)
### Q_a_s = t_board_X.q_network(states)
### print(Q_a_s)
### Q_a_s = tf.gather_nd(Q_a_s, tf.stack([tf.range(Q_a_s.shape[0]), tf.cast(actions, tf.int32)], axis=1))
### print(tf.stack([tf.range(Q_a_s.shape[0]), tf.cast(actions, tf.int32)], axis=1))
### print(Q_a_s)
### Q_a_s = t_board_X.target_q_network(states)
### print(Q_a_s)
### Q_a_s = tf.gather_nd(Q_a_s, tf.stack([tf.range(Q_a_s.shape[0]), tf.cast(actions, tf.int32)], axis=1))
### print(tf.stack([tf.range(Q_a_s.shape[0]), tf.cast(actions, tf.int32)], axis=1))
### print(Q_a_s)

# train_games(1, t_board_X, t_board_O)

print(t_board_X.next_states[1])
print(t_board_O.next_states[0])
