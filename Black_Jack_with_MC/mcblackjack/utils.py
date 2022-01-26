import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def transform_state(state):
    tr_state = list(state)
    tr_state[2] = int(tr_state[2])
    return tr_state

def generate_episode(env, agent):
    """
    Generate episode
    """
    
    s = env.reset()
    s = transform_state(s)
    action = agent.do_action(s)
    
    states = [s]
    rewards = []
    actions = [action]
    
    done = False
    
    while not done:

        s, reward, done, info = env.step(action)
        s = transform_state(s)
        action = agent.do_action(s)
        
        states.append(s)
        rewards.append(reward)
        actions.append(action)
    return states, rewards, actions


def mc_control_on_policy_first_visit(env, agent, gamma, n_episode):
    
    """
    Function learning on-policy agent by First Visit Monte Carlo
    
    @param env: environment
    @param agent: agent
    @param gamma: discounting factor
    @param n_episode: a number of episodes
    """
    
    for episode in range(n_episode):
        states_t, rewards_t, actions_t = generate_episode(env, agent)
        G = 0
        t = len(states_t) - 2
        
        for state_t, reward_t, action_t in zip(states_t[::-1][1:],
                                               rewards_t, 
                                               actions_t[::-1][1:]):
            
            G = gamma * G + reward_t
            
            if (state_t, action_t) not in list(zip(states_t, actions_t))[:t]:
                agent.update(G, state_t, action_t)
                
            t-=1

    return agent

def mc_control_on_policy_every_visit(env, agent, gamma, n_episode):

    """
    Function learning on-policy agent by Every Visit Monte Carlo
    
    @param env: environment
    @param agent: agent
    @param gamma: discounting factor
    @param n_episode: a number of episodes
    """
        
    for episode in range(n_episode):
        states_t, rewards_t, actions_t = generate_episode(env, agent)
        G = 0
        
        for state_t, reward_t, action_t in zip(reversed(states_t[:-1]),
                                               reversed(rewards_t), 
                                               reversed(actions_t[:-1])):
            
            G = gamma * G + reward_t
            
            agent.update(G, state_t, action_t)
                

    return agent

def mc_control_off_policy_first_visit(env, agent, gamma, n_episode):
    
    """
    Function learning off-policy agent by First Visit Monte Carlo
    
    @param env: environment
    @param agent: agent
    @param gamma: discounting factor
    @param n_episode: a number of episodes
    """ 
    
    for episode in range(n_episode):
        states_t, rewards_t, actions_t = generate_episode(env, agent)
        G = 0
        W = 1
        t = len(states_t) - 2
        for state_t, reward_t, action_t in zip(reversed(states_t[:-1]),
                                               reversed(rewards_t), 
                                               reversed(actions_t[:-1])):
            
            G = gamma * G + reward_t
            
            if (state_t, action_t) not in list(zip(states_t, actions_t))[:t]:
                agent.update(G, state_t, action_t, W)
                
            target_action = agent.target_policy(state_t)
            if action_t!=target_action:
                break
            _, probs = agent.research_policy(state_t)
            W *= 1/probs[action_t]   
            t-=1
            
    return agent

def mc_control_off_policy_every_visit(env, agent, gamma, n_episode):

    """
    Function learning off-policy agent by Every Visit Monte Carlo
    
    @param env: environment
    @param agent: agent
    @param gamma: discounting factor
    @param n_episode: a number of episodes
    """ 
    
    for episode in range(n_episode):
        states_t, rewards_t, actions_t = generate_episode(env, agent)
        G = 0
        W = 1
        for state_t, reward_t, action_t in zip(reversed(states_t[:-1]),
                                               reversed(rewards_t), 
                                               reversed(actions_t[:-1])):
            
            G = gamma * G + reward_t
            
            agent.update(G, state_t, action_t, W)
            target_action = agent.target_policy(state_t)
            if action_t!=target_action:
                break
            _, probs = agent.research_policy(state_t)
            W *= 1/probs[action_t]   

    return agent


def mc_control_per_decision_first_visit(env, agent, gamma, n_episode):
    
    """
    Modification of function learning off-policy agent by First Visit Monte Carlo
    with per-decision importance sampling
    
    @param env: environment
    @param agent: agent
    @param gamma: discounting factor
    @param n_episode: a number of episodes
    """     
    
    for episode in range(n_episode):
        states_t, rewards_t, actions_t = generate_episode(env, agent)
        G = 0
        W = 1
        t = len(states_t) - 2
        for state_t, reward_t, action_t in zip(reversed(states_t[:-1]),
                                               reversed(rewards_t), 
                                               reversed(actions_t[:-1])):
            
            G = gamma * W * G + reward_t
            
            if (state_t, action_t) not in list(zip(states_t, actions_t))[:t]:
                agent.update(G, state_t, action_t, 1)
                
            target_action = agent.target_policy(state_t)
            if action_t!=target_action:
                break
            _, probs = agent.research_policy(state_t)
            W = 1/probs[action_t]   
            t-=1
            
    return agent

def mc_control_per_decision_every_visit(env, agent, gamma, n_episode):

    """
    Modification of function learning off-policy agent by Every Visit Monte Carlo
    with per-decision importance sampling
    
    @param env: environment
    @param agent: agent
    @param gamma: discounting factor
    @param n_episode: a number of episodes
    """     
    
    for episode in range(n_episode):
        states_t, rewards_t, actions_t = generate_episode(env, agent)
        G = 0
        W = 1
        for state_t, reward_t, action_t in zip(reversed(states_t[:-1]),
                                               reversed(rewards_t), 
                                               reversed(actions_t[:-1])):
            
            G = gamma * W * G + reward_t
            
            agent.update(G, state_t, action_t, 1)
            target_action = agent.target_policy(state_t)
            if action_t!=target_action:
                break
            _, probs = agent.research_policy(state_t)
            W = 1/probs[action_t]   

    return agent


def plot_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.show()
    
    
def plot_blackjack_value(V):
    player_sum_range = range(12, 22)
    dealer_show_range = range(1, 11)
    X, Y = torch.meshgrid([torch.tensor(player_sum_range),
                           torch.tensor(dealer_show_range)])

    plot_surface(X, Y, V[:,:,0].numpy(),
                 "Value function without usable ace")
    plot_surface(X, Y, V[:,:,1].numpy(),
                 "Value function with usable ace")