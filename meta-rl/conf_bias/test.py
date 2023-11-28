import argparse
import getpass
from datetime import datetime
import json
import os 

import torch
from glob import glob
import sys

sys.path.append('/u/jschubert/learning_bias/meta-rl/conf_bias/src')

from agent import A2COptim
from environment import CompleteTask
from utils import AgentMemoryOptim, LogProgress

from runner import Runner

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='/u/jschubert/learning_bias/meta-rl/conf_bias/data/conf_bias/gpu_eps5000_bs64_agency_mask_policy_value_loss_20231019_16:30/') # set folder path to load params.json
    parser.add_argument('--test_eps', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=48)
    parser.add_argument('--agency_test', type=str, default='mask_policy_value_loss')

    
    args = parser.parse_known_args()[0]
    return args


def read_params(args):
    file_path = os.path.join(args.folder_path, 'params.json')

    with open(file_path, "r") as infile:
        data = json.load(infile)

    # Add items from dictionary to the args namespace
    for key, value in data.items():
        setattr(args, key, value)

def initialize_agent(args):

    if args.agent_model == 'Transformer':
        a2c = A2COptim(args.test_batch_size, 
                       args.hidden_size, 
                       args.input_size,  
                       args.learning_rate, 
                       args.is_cuda, 
                       num_layers = args.num_layers_transformer, 
                       n_heads = args.n_heads)

    elif args.agent_model == 'LSTM':
        a2c = A2COptim(args.test_batch_size, 
                       args.hidden_size, 
                       args.input_size, 
                       args.learning_rate, 
                       args.is_cuda)
                   
    return a2c

if __name__ == '__main__':
    """Argument parsing"""
    args = get_args()
    args.folder = args.folder_path

    read_params(args)
    #args.is_cuda = True if torch.cuda.is_available() and args.agent_model == 'Transformer' else False
    args.is_cuda = False

    # Initialize the bandit environment
    debug = True if args.exp_name == 'test' else False
    bandit = CompleteTask(batch_size=args.test_batch_size, 
                         train=False,
                         success=args.success, 
                         fail=args.fail,
                         debugging=debug)

    # initialize agent
    agent = initialize_agent(args)

    # initialize memory
    memory = AgentMemoryOptim(args.with_time, 
                              args.is_cuda,
                              args.return_coef,
                              args.value_coef,
                              args.return_fn,
                              args.agency_test)

    # configure logging
    logger = LogProgress(args.folder + '/test/', 
                        args.test_batch_size,
                        args.test_eps,
                        train=False,
                        plot_freq=20,
                        plot_window_size=5)

    # initialize training
    runner = Runner(agent,
                    bandit, 
                    memory,
                    logger)
    
    # start training
    print('start testing')

    runner.test(args.test_eps,
                args.folder)

    print('finished testing')