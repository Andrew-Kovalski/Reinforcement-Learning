import torch
from .base import MCBlackJackAgent

class MCBJDetermAgent(MCBlackJackAgent):
    
    def __init__(self, env, gamma, seed=0):
        
        """
        Agent for playing blackjack with deterministic policy
        """
        
        super().__init__(env, gamma, seed=seed)
        
        # deterministic policy
        self.policy = torch.randint(high=self.n_actions,
                                    size=(self.n_states_player, 
                                          self.n_states_dealer,
                                          self.n_states_ace),
                                    generator=self.generator)
                                 
        
    def func_policy(self, state):
        """
        Policy function from the current state
        
        state: current state
        """
        p, d, a = state
        return self.policy[p, d, a].item()
    
    def do_action(self, state):
        """
        To do action with current policy
        """
        return self.func_policy(state)
    
    def update(self, G, state, action):
        """
        Update Q-value and policy
        
        G: revenue
        """
        
        p, d, a = state
        self.N[p, d, a, action] += 1
        self.Q[p, d, a, action] += (1/self.N[p, d, a, action])*(G - self.Q[p, d, a, action])
        self.policy[p, d, a] = torch.argmax(self.Q[p, d, a, :]).item()    
        
class MCBJEpsGreedyOnPolicyAgent(MCBlackJackAgent):
    
    def __init__(self, env, gamma, eps, seed=0):
        
        """
        Initialize agent for playing blackjack with epsilon-greedy policy
        """
        super().__init__(env, gamma, seed=seed)
        
        self.policy = torch.rand(size=(self.n_states_player, 
                                       self.n_states_dealer,
                                       self.n_states_ace,
                                       self.n_actions),
                                 generator=self.generator)
        
        # probability of choosing a random action
        self.eps = eps
        
    def func_policy(self, state):
        p, d, a = state
        return torch.multinomial(self.policy[p, d, a], 1).item()
    
    def do_action(self, state):
        return self.func_policy(state)
    
    def update(self, G, state, action):
        
        p, d, ace = state
        self.N[p, d, ace, action] += 1
        self.Q[p, d, ace, action] += (1/self.N[p, d, ace, action])*(G - self.Q[p, d, ace, action])
        A = torch.argmax(self.Q[p, d, ace, :]).item()
        self.policy[p, d, ace, :] = self.eps/self.n_actions
        self.policy[p, d, ace, A] += 1 - self.eps
        
class MCBJEpsGreedyOffPolicyAgent(MCBlackJackAgent):
    
    def __init__(self, env, gamma, seed=0):
        
        """
        Initialize agent with off-policy learning
        """
        super().__init__(env, gamma, seed=seed)
        
    def func_policy(self, state):
        pass
    
    def target_policy(self, state):
        """
        Greedy policy for optimal Q-value 
        """
        p, d, a = state
        return torch.argmax(self.Q[p, d, a, :]).item()
        
    def research_policy(self, state):
        """
        Policy for importance sampling
        
        Return
        action and probabilities of choice all actions
        """
        p, d, a = state
        probability = torch.nn.Softmax()(self.Q[p, d, a, :])
        action = torch.multinomial(probability, 1).item()
        return action, probability
    
    def do_action(self, state, test=False):
        if test:
            return self.target_policy(state)
        else:
            action, _ = self.research_policy(state)
            return action
    
    def update(self, G, state, action, w):
        
        p, d, ace = state
        self.N[p, d, ace, action] += w
        self.Q[p, d, ace, action] += (w/self.N[p, d, ace, action])*(G - self.Q[p, d, ace, action])