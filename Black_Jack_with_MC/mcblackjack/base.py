import torch
from abc import ABC, abstractmethod


class MCBlackJackAgent:
    def __init__(self, env, gamma, seed=0):
        
        """
        Basic agent for playing blackjack
        
        @param env: environment
        @param gamma: discounting factor
        @param seed: number of pseudorandom sequence
        
        """
        
        self.gamma = gamma
        self.seed = seed
        
        # observation space for player
        self.n_states_player = env.observation_space[0].n
        
        # observation space for dealer
        self.n_states_dealer = env.observation_space[1].n
        
        # observation space for using ace
        self.n_states_ace = env.observation_space[2].n
        
        # number of actions
        self.n_actions = env.action_space.n
        
        # table Q-value function
        self.Q = torch.rand((self.n_states_player, 
                             self.n_states_dealer, 
                             self.n_states_ace, 
                             self.n_actions))
        
        # Array of the number of visits to the current state
        self.N = torch.zeros((self.n_states_player, 
                              self.n_states_dealer, 
                              self.n_states_ace, 
                              self.n_actions))
        
        # generator of a pseudorandom value
        self.generator = torch.Generator().manual_seed(self.seed)
    
    @abstractmethod
    def func_policy(self, state):
        raise NotImplementedError        
    
    @abstractmethod
    def do_action(self, state):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, G, state, action):
        raise NotImplementedError
    
    def get_Q(self):
        """
        Return table Q-value function
        """
        
        return self.Q