from torch.distributions import Categorical
from model import LSTM
from model import Transformer
from utils import AgentMemoryOptim
import torch.nn as nn
import torch
import torch.optim as optim

class A2COptim(nn.Module):
    def __init__(self, batch_size, hidden_size, input_size, lr, is_cuda, num_layers=None, n_heads=None) -> None:
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

        self.agent_model = 'LSTM' if n_heads is None else 'Transformer'
        self.num_actions = 2

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size

        if self.agent_model == 'Transformer':
            self.net = Transformer(batch_size, input_size, hidden_size, is_cuda, num_layers, n_heads)
        elif self.agent_model == 'LSTM':
            self.net = LSTM(batch_size, hidden_size, input_size, is_cuda)

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
        
        # replace policy with 0,1 if nan
        #mask_nan = torch.isnan(policy).any(axis=1)
        #if self.is_cuda:
        #    policy = torch.where(mask_nan.unsqueeze(1), torch.tensor([0.0, 1.0]).cuda(), policy)
        #else:
        #    policy = torch.where(mask_nan.unsqueeze(1), torch.tensor([0.0, 1.0]), policy)
        #policy[mask_nan, :] = torch.tensor([0.0, 1.0])
        
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