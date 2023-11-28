import argparse
import getpass
from datetime import datetime
import json
import os 

import torch
from glob import glob
import sys

sys.path.append('/u/jschubert/learning_bias/meta-rl/agency/src')

from agent import A2COptim
from environment import PartialTask
from utils import AgentMemoryOptim, LogProgress

from runner import Runner

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='/u/jschubert/learning_bias/meta-rl/agency/data/test/test_20230922_09:09') # set folder path to load params.json
    parser.add_argument('--test_eps', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=48)
    parser.add_argument('--agency_test', type=str, default='mask_policy_value_loss')
    parser.add_argument('--loss_fn', type=str, default='')
    # 4 * 18 = 72 -> matching amount of human participant simulations
    
    
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

    read_params(args)
    #args.is_cuda = True if torch.cuda.is_available() and args.agent_model == 'Transformer' else False
    args.is_cuda = False

    # Initialize the bandit environment
    testing = True if args.exp_name == 'test' else False
    bandit = PartialTask(args.test_batch_size, 
                         args.success, 
                         args.fail,
                         testing=testing)

    # initialize agent
    agent = initialize_agent(args)

    # initialize memory
    memory = AgentMemoryOptim(args.feedback, 
                            args.with_time, 
                            args.is_cuda,
                            args.return_coef,
                            args.value_coef,
                            args.return_fn,
                            args.agency_test)

    # configure logging
    folder_path = args.folder_path + '/test/'
    logger = LogProgress(folder_path, 
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
                args.folder_path)

    print('finished testing')