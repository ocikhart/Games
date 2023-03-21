import numpy as np
from itertools import product
import pandas as pd
import random

#For BASE3 indexing
BASE3 = [6561, 2187, 729, 243, 81, 27, 9, 3, 1]
STATE_SHAPE = 9
BOARD_SHAPE = (3,3)
MAX_INDEX = 19683

class TicTacToe():
    def __init__(self,player = 1,reward_type ='goal_reward'):
        '''
        player: Role agent should play. If X, agent has the first turn else agent has second turn
        reward_type: 'goal_reward' or 'action_penalty'
        '''
        self.board = np.array([0]*STATE_SHAPE).reshape(BOARD_SHAPE)
        self.reward_type = reward_type
        self.winning_sequence = None #Keep track of winning move made by agent
        self.first_move = None #Keep track of first move made by agent
        if player == 1:
            self.me = 1
            self.id = 1
            self.opponent = 2
        else:
            self.me = 2
            self.id = 2
            self.opponent = 1
     
        self.game_over = False #Flag indicating whether game is over
        # Mapping of action representation in board to action representation in tuple 
        self.b_to_s = {'__':0,'X':1,'O':2} 
        # Mapping of action representation in tuple to action representation in board
        self.s_to_b = {0:'__',1:'X',2:'O'} 
        
        #Create mapping from 2D position in board to 1D position in tuple
        positions = self.available_positions()
        self.b2_to_s1 = {position:i for (i,position) in enumerate(positions)}
        
        #Create mapping from 1D position in tuple to 2D position in board 
        self.s1_to_b2 = {i:position for (i,position) in enumerate(positions)}
        
        #State the current player is in
        self.starting_state = self.board
        
        #Initialize all possible states of the game
        l_o_l = [list(range(3)) for _ in range(STATE_SHAPE)]
        states = list(product(*l_o_l))
        
        #Player X states include states with odd number of blanks and both players have occupied equal number of slots
        #Player O players after Player X, so player O states include states with even number of blanks and where
        #player X has occupied one more slot than player O
        playerX_states = [np.array(state) for state in states if (state.count(0)%2 == 1 and state.count(1)==state.count(2))]
        playerO_states = [np.array(state) for state in states if (state.count(0)%2 == 0 and state.count(1)==(state.count(2)+1))]
        
        #States 
        #self.board_full_states = {state for state in states if state.count(0)==0}
        if player == 1:
            self.my_states = playerO_states
        else:
            self.my_states = playerX_states
        
        #Generate next states (move N+1)
        self.next_states = [[] for _ in range(MAX_INDEX)]
        self.next_states_p = [[] for _ in range(MAX_INDEX)]
        #tmp_states = self.my_states.copy()
        for s1 in self.my_states:
            if self.is_win(s1.reshape(BOARD_SHAPE),self.me):
                continue
            j = self.state_index(s1)
            actions = [i for i,x  in np.ndenumerate(s1) if x == 0]
            nsj1 = [np.copy(s1) for _ in range(len(actions))]
            for i,action in enumerate(actions):
                nsj1[i][action] = self.opponent
            losing_state = False
            for s2 in nsj1:
                if self.is_win(s2.reshape(BOARD_SHAPE),self.opponent):
                    losing_state = True
                    continue
                actions = [i for i,x  in enumerate(s2) if x == 0]
                nsj2 = [np.copy(s2) for _ in range(len(actions))]
                for i,action in enumerate(actions):
                    nsj2[i][action] = self.me
                self.next_states[j] += nsj2
            if losing_state:
                #self.next_states_p[j] = [0.2/max(1,len(self.next_states[j])) for _ in range(len(self.next_states[j]))]
                self.next_states_p[j] = [0.2/max(1,len(nsj1)) for _ in range(len(self.next_states[j]))]
            else:
                #self.next_states_p[j] = [1/max(1,len(self.next_states[j])) for _ in range(len(self.next_states[j]))]
                self.next_states_p[j] = [1/max(1,len(nsj1)) for _ in range(len(self.next_states[j]))]
            self.next_states[j] = np.asarray(self.next_states[j])
            self.next_states_p[j] = np.asarray(self.next_states_p[j])
        
        #Experiences
        self.experiences = []
          
    
    def reset_board(self):
        "Function to reset game and reset board to starting state"
        self.board = np.array([0]*STATE_SHAPE).reshape(BOARD_SHAPE)
        self.starting_state = self.board
        self.game_over = False
        self.winning_sequence = None
        self.first_move = None
    
    def show_board(self):    
        "Shows board as a pandas dataframe"
        return pd.DataFrame(self.board)
    
    def board_to_state(self):
        "Convert a board to a state in tuple format"
        return tuple([self.b_to_s[x] for x in np.ravel(self.board)])
    
    @staticmethod
    def state_index(state):
        "Returns index of state"
        return int(np.dot(BASE3, state))
    
    @staticmethod
    def possible_actions(state):
        "Return possible actions given a state"
        return [i for i,x  in enumerate(np.ravel(state)) if x ==0]
    
    def next_possible_states(self,state):
        "Return array of possible next states for a given a state"
        return self.next_states[np.dot(BASE3, np.ravel(state))]
        
    def is_game_over(self):
        "Function to check if game is over"
        if not np.any(self.board == 0) :
            self.game_over = True
            
        return self.game_over
    
    def available_positions(self):
        "Return available positions on the board"
        x,y = np.where(self.board == 0)
        return[(x,y) for x,y in zip(x,y)]
    
    
    def win(self,player):
        "Check if player won the game and record the winning sequence"
        if np.all(self.board[0,:] == player):
            self.winning_sequence = 'R1'
        elif np.all(self.board[1,:] == player): 
            self.winning_sequence = 'R2'
        elif np.all(self.board[2,:] == player):
            self.winning_sequence = 'R3'
        elif np.all(self.board[:,0] == player):
            self.winning_sequence = 'C1'
        elif np.all(self.board[:,1] == player):
            self.winning_sequence = 'C2'
        elif np.all(self.board[:,2] == player):
            self.winning_sequence = 'C3'
        elif np.all(self.board.diagonal()==player):
            self.winning_sequence = 'D1'
        elif  np.all(np.fliplr(self.board).diagonal()==player):
            self.winning_sequence = 'D2'
        else:
            return False
        
        return True
    
    def is_win(self,state,player):
        "Check if state of a player is a win"
        row_win = np.all(state[0,:] == player) or np.all(state[1,:] == player) or np.all(state[2,:] == player)
        col_win = np.all(state[:,0] == player) or np.all(state[:,1] == player) or np.all(state[:,2] == player)
        diag_win = np.all(state.diagonal()==player) or np.all(np.fliplr(state).diagonal()==player)
        if row_win or col_win or diag_win:
            return True
        return False
    
    
    def my_move(self,position):
        "Fills out the board in the given position with the action of the agent"
        
        assert position[0] >= 0 and position[0] <= 2 and position[1] >= 0 and position[1] <= 2 , "incorrect position"
        assert self.board[position] == 0 , "position already filled"
        assert np.any(self.board == 0) , "Board is complete"
        assert self.win(self.me) == False and self.win(self.opponent)== False , " Game has already been won"
        self.board[position] = self.me
        
        I_win = self.win(self.me)
        
        if self.reward_type == 'goal_reward':
            if I_win:
                self.game_over = True
                return 1
            else:
                return 0
            
        elif self.reward_type == 'action_penalty':
            if I_win:
                self.game_over = True
                return 0
            else:
                return -1
    
    def opponent_move(self,position):
        "Fills out the board in the given position with the action of the opponent"
        assert position[0] >= 0 and position[0] <= 2 and position[1] >= 0 and position[1] <= 2 , "incorrect position"
        assert self.board[position] == 0 , "position already filled"
        assert np.any(self.board == 0) , "Board is complete"
        assert self.win(self.me) == False and self.win(self.opponent)== False , " Game has already been won"
        self.board[position] = self.opponent
        
        opponent_win = self.win(self.opponent)
        
        if self.reward_type == 'goal_reward':
            if opponent_win:
                self.game_over = True
                return -1
            
            else:
                return 0
            
        elif self.reward_type == 'action_penalty':
            if opponent_win:
                self.game_over = True
                return -10
            
            else:
                return -1
            
    
    def pick_best_action(self,action_type,eps=None):
        '''Given a Q function return optimal action
        If action_type is 'greedy' return best action with ties broken randomly else return epsilon greedy action
        '''
        #Get possible actions
        actions =  self.possible_actions(self.board)
        states = np.tile(np.ravel(self.board), (len(actions),1))
        for i,action in enumerate(actions):
            states[i][action] = self.me
        best_action = []
        best_action_value = -np.Inf
        
        Q_s_a = self.Q(states)
        for i,action in enumerate(actions):            
            if Q_s_a[i] == best_action_value:
                best_action.append(action)
            elif Q_s_a[i] > best_action_value:
                best_action = [action]
                best_action_value = Q_s_a[i]
        best_action = random.choice(best_action)

        if action_type == 'greedy':
            return self.s1_to_b2[best_action]
        else:
            assert eps != None , "Include epsilon parameter"
            n_actions =len(actions) #No of legal actions 
            p = np.full(n_actions,eps/n_actions)
            #Get index of best action
            best_action_i = actions.index(best_action)
            p[best_action_i]+= 1 - eps
            return self.s1_to_b2[np.random.choice(actions,p=p)]
        
    def Q(self,states):
        return np.array([0.]*len(states))
