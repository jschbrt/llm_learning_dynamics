from torch.distributions import Categorical
from model import LSTM
from model import Transformer
from utils import AgentMemory, AgentMemoryOptim
import torch.nn as nn
import torch.optim as optim

class A2C(nn.Module):
    def __init__(self, batch_size, hidden_size, input_size, lr, trials, is_cuda, writer=None, num_actions=2, agent_model='None') -> None:
        """
        Implementation of the Advantage Actor-Critic (A2C) Network

        Args:
            :hidden_size: Hidden Size of the LSTMCell
            :lr: Learning rate for the optimization of the A2C.
            :trials: number of trials per rollout for creating the memory
            :is_cuda: for putting net on GPU or not 
            :num_actions: The amount of actions the actor can utilize
        """ 
        super().__init__()
        self.agent_model = agent_model
        self.num_actions = num_actions
        self.net = Transformer(batch_size, hidden_size, input_size, is_cuda, writer=writer)
        self.memory = AgentMemory(is_cuda, writer)

        self.is_cuda = is_cuda
        if self.is_cuda:
            self.net.cuda()

        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), self.lr)
    
    def step(self, x):
        """ 
        The step function when a step in the environment is made.

        Args:
            x: torch.tensor(dim=4) eg.[0,1]: one_hot for prev_action 2, prev_reward: 1, time: 3 -> [0,1,1,3]

        Returns:
            a: the next action as a single torch value (0 or 1)
            log_policy_a: the log probability of the policy for action a
            entropy: the entropy of the policy ?? TODO: Why is entropy calculated for both values? 
            baseline: the baseline or value function for action a 
        """

        """Evaluate the A2C by forwarding the input to the LSTM"""
        policy, baseline = self.net.forward(x)

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical distribution represented by these probabilities
        cat = Categorical(policy)
        a = cat.sample()
        log_policy_a = cat.log_prob(a)
        entropy = cat.entropy()  # -sum(log probabilities [aka logits] * probs)

        return (a, log_policy_a, entropy, baseline)
    
    def reset(self):
        """
        Resets the hidden state of the LSTM at the beginning of an episode.
        """
        self.net.init_hidden(self.is_cuda)

"""
Simple test

import torch
agent = A2C(...)

x = [0,1,1,3] # one_hot for action 2 -> [0,1], reward: 1, time: 3 -> [0,1,1,3]
x = torch.FloatTensor(x)
# no need for it due to LSTMCell x = x.view(1,-1) # append the sequence dimension
a, log_policy_a, entropy, baseline = agent.step(x)

"""


class A2COptim(nn.Module):
    def __init__(self, batch_size, hidden_size, input_size, num_layers, n_heads, lr, is_cuda, writer=None, num_actions=2, agent_model='None') -> None:
        """
        Implementation of the Advantage Actor-Critic (A2C) Network

        Args:
            :hidden_size: Hidden Size of the LSTMCell
            :lr: Learning rate for the optimization of the A2C.
            :trials: number of trials per rollout for creating the memory
            :is_cuda: for putting net on GPU or not 
            :num_actions: The amount of actions the actor can utilize
        """ 
        super().__init__()

        self.agent_model = agent_model
        self.num_actions = num_actions
        self.n_heads = n_heads
        if agent_model == 'Transformer':
            self.net = Transformer(batch_size, input_size, hidden_size, is_cuda, num_layers, n_heads, writer)
        elif agent_model == 'LSTM':
            self.net = LSTM(batch_size, hidden_size, input_size, is_cuda, writer)
        self.memory = AgentMemoryOptim(is_cuda, writer)

        self.is_cuda = is_cuda
        if self.is_cuda:
            self.net.cuda()

        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), self.lr)
    
    def step(self, x, idx):
        """ 
        The step function when a step in the environment is made.

        Args:
            x: torch.tensor(dim=7) -> [one_hot(current_cue), one_hot(prev_action), prev_reward]
        Returns:
            a: the next action as a single torch value (0 or 1)
            log_policy_a: the log probability of the policy for action a
            entropy: the entropy of the policy
            baseline: the baseline or value function for action a 
        """

        """Evaluate the A2C by forwarding the input to the LSTM"""
        if self.agent_model == 'LSTM':
            policy, baseline = self.net.forward(x)
        else:
            policy, baseline = self.net.forward(x, idx)
        
        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical distribution represented by these probabilities
        cat = Categorical(policy)
        a = cat.sample()
        log_policy_a = cat.log_prob(a)
        entropy = cat.entropy()  # -sum(log probabilities [aka logits] * probs)

        return (a, log_policy_a, entropy, baseline, policy)
    
    def reset(self):
        """
        Resets the hidden state of the LSTM at the beginning of an episode.
        """
        if self.agent_model == 'LSTM':
            self.net.init_hidden(self.is_cuda)
        else:
            pass