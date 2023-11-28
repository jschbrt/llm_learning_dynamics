"""Script for testing the agent on the environment."""

import argparse
import getpass
from datetime import datetime
import json
import os 
from glob import glob

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from agent import A2COptim
from training_optimBias import Runner
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='transformer_uniform_distribution')
    parser.add_argument('--hidden_size', type=int, default=192)
    parser.add_argument('--return_coef', type=float, default=0.8)
    parser.add_argument('--entropy_final_value', type=float, default=0)
    parser.add_argument('--loss_fn', type=str, default='discounted_return') # not needed, because only during trainng used
    parser.add_argument('--with_time', type=str, default='false')
    parser.add_argument('--env_mode', type=str, default='test')
    parser.add_argument('--with_counter_reward', type=int, default=0) #0 normal reward, 1 counter reward, 2, both rewards
    parser.add_argument('--agent_model', type=str, default='Transformer')
    parser.add_argument('--num_layers_transformer', type=int, default=2)

    parser.add_argument('--task-id', type=int, default=0)

    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    """Argument parsing"""
    args = get_args()

    """Hyperparameters from the paper""" 
    is_cuda = True if torch.cuda.is_available() else False
    episode_length = 96 # actually modelled via environment
    learning_rate = 0.003
    input_size = 7 # cue(one_hot [4]), prev_action (one_hot [2]), prev_reward 

    if args.with_time == 'true':
        input_size += 1
        with_time = True
    else:
        with_time = False

    if args.with_counter_reward == 2:
        input_size += 1 # add dim if we concat prev_reward and counter_reward
    
    max_grad_norm = 50
    value_coef = 0.5 # beta_v


    # model_name for tests where discount and entropy final was varied
    #model_name = f'RL__rc_{args.return_coef}__efv_{args.entropy_final_value}__env_mode_{nme}__id_{args.task_id}'
    #model_name = f'RL__rc_{args.return_coef}__efv_{args.entropy_final_value}w__rew_{args.with_counter_reward}__id_{args.task_id}'
    #model_name = f'RL__rc_0.85__efv_0w__rew_0__id_{args.task_id}'
    # model_name for tests where hidden_size is changed
    #model_name = f'RL__hs_{args.hidden_size}__id_{args.task_id}'
    model_name = f'Transformer__150000__mode_train__num_layers_2__linear_size_192'


    folder_path = f'../../../../eris/scratch/jschubert/training/{args.exp_name}/{model_name}/'

    if args.task_id == 0:
        with_random = True
    else:
        with_random = False
    
    writer = SummaryWriter(log_dir=folder_path+'tb/test/')

    # function for testing on random environment
    test_random = False
    if  test_random == True:
        #folder_path = f'../../../../eris/scratch/jschubert/training/test_compare_with_rw/'
        batch_size = 2
        test_eps = 100
        test_case = args.env_mode

        a2c = A2COptim(batch_size, args.hidden_size, input_size, learning_rate, is_cuda, writer)
        runner = Runner(episode_length, batch_size, folder_path,a2c,writer, with_time, with_random, args.with_counter_reward)
        runner.test(test_eps, test_case, test_learning_stages=False)

    # function for testing on participant data (50 participants)
    else:
        batch_size = 1
        test_eps = 1
        test_case = args.env_mode.replace('test','exp1')
        num_participants = 50
        # Test if parameters correspond to each other
        """
        params = {'is_cuda': is_cuda,
                'task_id': args.task_id,
                'trials_per_eps': episode_length,
                'learning_rate': learning_rate,
                'hidden_size': args.hidden_size,
                'input_size': input_size,
                'max_grad_norm': max_grad_norm,
                'overfit': False,
                'return_coef': args.return_coef,
                'value_coef': value_coef,
                'entropy_final_value': args.entropy_final_value,
                'with_time': args.with_time,
                'with_random':with_random,
                'loss_fn':args.loss_fn,
                #'with_counter_reward':args.with_counter_reward}
                }
        
        current_json = json.dumps(params)
        with open(folder_path+"params.json", "r") as outfile:
            former_json = json.load(outfile)
        del former_json['date_time']
        del former_json['train_eps']
        del former_json['batch_size']
        del former_json['env_mode']
        #del former_json['loss_fn']
        if not sorted(former_json.items()) == sorted(params.items()):
            print(params.items())
            print('\nFormer:\n')
            print(former_json.items())
            raise ValueError('The parameter configurations differ.')
        """
        a2c = A2COptim(batch_size, args.hidden_size, input_size, args.num_layers_transformer, learning_rate, is_cuda, writer, agent_model=args.agent_model)

        bar = tqdm(total=50)
        for i in range(num_participants):
            runner = Runner(episode_length, batch_size, folder_path,a2c,writer, with_time, with_random, args.with_counter_reward)
            runner.test(test_eps, test_case, p_idx=i, test_learning_stages=False)
            bar.update(1)
