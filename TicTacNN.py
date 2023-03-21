import numpy as np
import TicTacToe as ttt
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.optimizers import adam_v2 as Adam

INPUT = 9           # size of state
GAMMA = 0.995       # discount factor
ALPHA = 1e-3        # learning rate  
TAU = 500*1e-3          # Soft update parameter.


class TicTacNN(ttt.TicTacToe):
    def __init__(self,player = 1,reward_type ='goal_reward'):
        '''
        player: Role agent should play. If X, agent has the first turn else agent has second turn
        reward_type: 'goal_reward' or 'action_penalty'
        '''
        ttt.TicTacToe.__init__(self,player,reward_type)
        
        # Create the Q-Network
        self.q_network = Sequential(
            [
                Dense(32, input_dim=INPUT, activation = 'relu'),
                Dense(32, activation = 'relu'),
                Dense(1, activation = 'linear')
            ]
        )

        # Create the target Q^-Network
        self.target_q_network = Sequential(
            [
                Dense(32, input_dim=INPUT, activation = 'relu'),
                Dense(32, activation = 'relu'),
                Dense(1, activation = 'linear')
            ]
        )
        
        # Create optimizer Adam
        self.optimizer = Adam.Adam(ALPHA)

    def Q(self,states):
        return self.q_network(states)
    
    def update_target_network(self):
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
            self.target_q_network.weights, self.q_network.weights
        ):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

