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

def unpack_experiences(player):
    """ 
    Unpacks player.experiences and returns tensors for custom training loop
    """
    e = list(map(list, zip(*player.experiences)))
    states = np.asarray(e[0],dtype=np.float32)
    rewards = np.asarray(e[1],dtype=np.float32)
    done_vals = np.asarray(e[2],dtype=np.float32)
    next_states_idx = []
    next_states = [[] for _ in range(len(states))]
    next_states_p = [[] for _ in range(len(states))]
    for i, s in enumerate(states):
        idx = ttr.TicTacToe.state_index(s)
        if np.size(player.next_states[idx]) == 0:
            continue
        next_states_idx.append(i)
        next_states[i] = tf.convert_to_tensor(player.next_states[idx],dtype=tf.float32)
        next_states_p[i] = tf.convert_to_tensor(player.next_states_p[idx],dtype=tf.float32)
    states = tf.convert_to_tensor(states,dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards,dtype=tf.float32)
    done_vals = tf.convert_to_tensor(done_vals,dtype=tf.float32)
    
    return states, rewards, next_states_idx, next_states, next_states_p, done_vals

def compute_loss(player, length, states, rewards, next_states_idx, next_states, next_states_p, done_vals, gamma):
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
    # Compute max Q^(s,a)
    max_qsa = [tf.constant(-1.0) for _ in range(length)]
    # Get the q_values
    q_values = tf.reshape(player.q_network(states), [-1])
    for i in next_states_idx:
        qsa = tf.reduce_max(tf.reshape(player.target_q_network(next_states[i]),[-1])*next_states_p[i], axis=-1)
        if qsa > max_qsa[i]:
            max_qsa[i] = qsa
    max_qsa = tf.convert_to_tensor(max_qsa)
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * gamma * max_qsa
    # Compute the loss
    loss = MSE(y_targets, q_values)  
    return loss

@tf.function
def player_learn(player, length, states, rewards, next_states_idx, next_states, next_states_p, done_vals, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """ 
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(player, length, states, rewards, next_states_idx, next_states, next_states_p, done_vals, gamma)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, player.q_network.trainable_variables)
    
    # Update the weights of the q_network.
    player.optimizer.apply_gradients(zip(gradients, player.q_network.trainable_variables))

    # update the weights of target q_network
    player.update_target_network()
    
    return loss


def play_games(n, player_X, player_O, silent = True):
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
            if not silent:
                print(player_X.show_board())
                print("-----------------------------------")
            board_X = np.copy(player_X.board)
            board_O = np.copy(player_O.board)
            reward_O = player_O.opponent_move(action_X)
            if player_X.win(player_X.me):
                if not silent:
                    print(f"WINNING MOVE of player {player_X.me} ---> Winning Sequence {player_X.winning_sequence}")
                wins_X += 1
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if not np.any(player_X.board == 0):
                if not silent:
                    print(f"DRAW")
                draws += 1
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if moved_O:
                player_O.experiences.append([np.ravel(board_O), reward_O, 0])
            
            action_O = player_O.pick_best_action('not_greedy', 0.2)
            reward_O = player_O.my_move(action_O)
            if not silent:
                print(player_O.show_board())
                print("-----------------------------------")
            board_O = np.copy(player_O.board)
            board_X = np.copy(player_X.board)
            reward_X = player_X.opponent_move(action_O)
            moved_O = True
            if player_O.win(player_O.me):
                if not silent:
                    print(f"WINNING MOVE of player {player_O.me} ---> Winning Sequence {player_O.winning_sequence}")
                wins_O += 1
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            if not np.any(player_O.board == 0):
                if not silent:
                    print(f"DRAW")
                draws += 1
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            player_X.experiences.append([np.ravel(board_X), reward_X, 0])
            i += 1
    return wins_X, draws, wins_O

def train_games(n, player_X, player_O, type_X='greedy', type_Y='greedy', randomize=0):
    """
    Simulates N games and for the player_X and player_O
    
    Args:
      n (int):                  Play N=n games
      player_X (TicTacToe):     Player X
      player_O (TicTacToe):     Player O
      randomize (int):          First X rounds are random   
    """
    wins_X = 0
    wins_O = 0
    draws = 0
    player_X.experiences = []
    player_O.experiences = []
    for j in range(n):
        player_X.reset_board()
        player_O.reset_board()
        i = 1
        first_O = False
        while True:
            if i <= randomize:
                action_X = player_X.pick_rnd_action()
            else:
                action_X = player_X.pick_best_action(type_X, eps=EPS)
            reward_X = player_X.my_move(action_X)
            board_X = np.copy(player_X.board)
            board_O = np.copy(player_O.board)
            reward_O = player_O.opponent_move(action_X)
            if player_X.win(player_X.me):
                wins_X += 1
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if not np.any(player_X.board == 0):
                draws +=1
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                break
            if first_O:
                player_O.experiences.append([np.ravel(board_O), reward_O, 0])
            
            if i <= randomize:
                action_O = player_O.pick_rnd_action()
            else:
                action_O = player_O.pick_best_action(type_Y, eps=EPS)
            reward_O = player_O.my_move(action_O)
            board_O = np.copy(player_O.board)
            board_X = np.copy(player_X.board)
            reward_X = player_X.opponent_move(action_O)
            first_O = True
            if player_O.win(player_O.me):
                wins_O += 1
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            if not np.any(player_O.board == 0):
                draws +=1
                player_O.experiences.append([np.ravel(board_O), reward_O, 1])
                player_X.experiences.append([np.ravel(board_X), reward_X, 1])
                break
            player_X.experiences.append([np.ravel(board_X), reward_X, 0])
            i += 1
    return wins_X, draws, wins_O

def train_X(opponent):
    loss = [0 for _ in range(EPOCHS)]
    wins_X = [0 for _ in range(EPOCHS)]
    draws = [0 for _ in range(EPOCHS)]
    wins_O = [0 for _ in range(EPOCHS)]
    x = np.arange(EPOCHS)
    for i in range(EPOCHS):
        wx, d, wo = train_games(GAME_BATCH, t_board_XNN, opponent, 'not_greedy', 'not_greedy', 0)
        wins_X[i] = wx
        draws[i] = d
        wins_O[i] = wo
        print(f"{wins_X[i]} : {draws[i]} : {wins_O[i]}")
        states, rewards, next_states_idx, next_states, next_states_p, done_vals = unpack_experiences(t_board_XNN)
        loss[i] = float(player_learn(t_board_XNN, len(states), states, rewards, next_states_idx, next_states, next_states_p, done_vals, ttn.GAMMA))
        print(f"Epoch {i+1} Loss = {loss[i]}")
        print("===================================")
    #saving the X model
    t_board_XNN.save_models(X_NN_file)
    #display results
    results_y = np.vstack([wins_X, draws, wins_O])
    ax[0].plot(x, np.asarray(loss))
    ax[1].stackplot(x, results_y)

def train_O(opponent):
    loss = [0 for _ in range(EPOCHS)]
    wins_X = [0 for _ in range(EPOCHS)]
    draws = [0 for _ in range(EPOCHS)]
    wins_O = [0 for _ in range(EPOCHS)]
    x = np.arange(EPOCHS)
    for i in range(EPOCHS):
        wx, d, wo = train_games(GAME_BATCH, opponent, t_board_ONN, 'not_greedy', 'not_greedy', 0)
        wins_X[i] = wx
        draws[i] = d
        wins_O[i] = wo
        print(f"{wins_X[i]} : {draws[i]} : {wins_O[i]}")
        states, rewards, next_states_idx, next_states, next_states_p, done_vals = unpack_experiences(t_board_ONN)
        loss[i] = float(player_learn(t_board_ONN, len(states), states, rewards, next_states_idx, next_states, next_states_p, done_vals, ttn.GAMMA))
        print(f"Epoch {i+1} Loss = {loss[i]}")
        print("===================================")
    #saving the O model
    t_board_ONN.save_models(O_NN_file)
    #display results
    results_y = np.vstack([wins_X, draws, wins_O])
    ax[0].plot(x, np.asarray(loss))
    ax[1].stackplot(x, results_y)

def play_XO(epochs, game_batch, silent=True):
    loss = [0 for _ in range(epochs)]
    wins_X = [0 for _ in range(epochs)]
    draws = [0 for _ in range(epochs)]
    wins_O = [0 for _ in range(epochs)]
    x = np.arange(epochs)
    for i in range(epochs):
        wx, d, wo = play_games(game_batch, t_board_XRND, t_board_ORND, silent)
        wins_X[i] = wx
        draws[i] = d
        wins_O[i] = wo
        print(f"{wins_X[i]} : {draws[i]} : {wins_O[i]}")
        print(f"Epoch {i+1} Loss = {loss[i]}")
        print("===================================")
    #display results
    results_y = np.vstack([wins_X, draws, wins_O])
    ax[0].plot(x, np.asarray(loss))
    ax[1].stackplot(x, results_y)



INPUT = 9
GAME_BATCH = 64
EPOCHS = 4
GAMMA = 0.995             # discount factor
SEED = 0  # Seed for the pseudo-random number generator.
EPS = 0.3 # ε for the ε-greedy policy.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.
training_directory_path = "/Users/ondrejcikhart/Desktop/Projects/Games/training/"
X_NN_file = training_directory_path + "MODEL64_AP"
O_NN_file = training_directory_path + "MODEL64_AP"

#tf.random.set_seed(SEED)

t_board_XNN = ttn.TicTacNN(player = 1,reward_type ='goal_reward')
t_board_XNN.load_models(X_NN_file)
t_board_ONN = ttn.TicTacNN(player = 2,reward_type ='goal_reward')
t_board_ONN.load_models(O_NN_file)
t_board_XRND = ttr.TicTacToe(player = 1,reward_type ='goal_reward')
t_board_ORND = ttr.TicTacToe(player = 2,reward_type ='goal_reward')

#Init display
plt.style.use('deeplearning.mplstyle')
fig, ax = plt.subplots(2)

train_X(t_board_ORND)
#train_O(t_board_XRND)
#play_XO(1, 256, True)

plt.show()