import numpy as np
from itertools import product
import pandas as pd
import tensorflow as tf
import TicTacToe as ttt
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.optimizers import adam_v2 as Adam

   
def Q_NN(state, action, player):
    print(state)
    assert action in range(len(state[0])), "incorrect position"
    assert state[0][action] == 0, "position already filled"
    assert np.any(state[0] == 0), "Board is complete"
    state[0][action] = player
    q = q_network(state)
    print(float(q[0][0]))
    return float(q[0][0])


def update_target_network(q_network, target_q_network):
    """
    Updates the weights of the target Q-Network using a soft update.
    
    The weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network.
    
    Args:
        q_network (tf.keras.Sequential): 
            The Q-Network. 
        target_q_network (tf.keras.Sequential):
            The Target Q-Network.
    """
    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa
    
    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    loss = MSE(y_targets, q_values) 
    
    return loss

@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """ 
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    update_target_network(q_network, target_q_network)


def play_games(n, player_X, player_O):
    """
    Simulates N games
    
    Args:
      n (int):                  Play N=n games
      player_X (TicTacToe):     Player X
      Q_X (TicTacToe):          Function Q for Player X
      player_O (TicTacToe):     Player O
      Q_O (TicTacToe):          Function Q for Player O
      
    Returns:
      p (scalar):  prediction
    """
    for j in range(n):
        game_over = False
        ttt.TicTacToe.reset_board(player_X)
        ttt.TicTacToe.reset_board(player_O)
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


INPUT = 9
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.


t_board_X = ttt.TicTacToe(player = 1,reward_type ='goal_reward')
t_board_O = ttt.TicTacToe(player = 2,reward_type ='goal_reward')

tf.random.set_seed(SEED)
# Create the Q-Network
q_network = Sequential([
    ### InputLayer(input_shape=INPUT),
    Dense(32, input_dim=INPUT, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(1, activation = 'linear')
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### InputLayer(input_shape=INPUT),
    Dense(32, input_dim=INPUT, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(1, activation = 'linear')
    ])

optimizer = Adam.Adam(ALPHA)
# y = q_network(np.array([(1,0,0,1,0,0,1,0,0)]))
actions = [i for i,x  in enumerate(np.ravel(t_board_X.board)) if x ==0]
print(actions)
states = np.tile(np.ravel(t_board_X.board), (len(actions),1))
print(states)
for i,action in enumerate(actions):
    states[i][action] = 1
print(states)
print(t_board_X.Q(states))

play_games(2, t_board_X, t_board_O)
