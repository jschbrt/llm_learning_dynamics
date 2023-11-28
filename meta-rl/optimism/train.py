"""Script to train the agent."""

import argparse
import getpass
from datetime import datetime
import json
import os 

import torch
from torch.utils.tensorboard.writer import SummaryWriter

import sys
sys.path.append('./src')

from agent import A2COptim
from training_optimBias import Runner
import cProfile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--train_eps', type=int, default=10)

    parser.add_argument('--hidden_size', type=int, default=-1)
    parser.add_argument('--num_layers_transformer', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--return_coef', type=float, default=0.8)
    parser.add_argument('--entropy_final_value', type=float, default=0)
    parser.add_argument('--loss_fn', type=str, default='discounted_return')
    parser.add_argument('--with_time', type=str, default='true')
    parser.add_argument('--env_mode', type=str, default='train')
    parser.add_argument('--success', type=int, default=0.5)
    parser.add_argument('--fail', type=int, default=0)
    parser.add_argument('--with_counter_reward', type=int, default=0) #0 normal reward, 1 counter reward, 2, both rewards
    parser.add_argument('--agent_model', type=str, default='Transformer')

    parser.add_argument('--task-id', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    """Argument parsing"""
    args = get_args()
    print('start training')
    """Hyperparameters from the paper""" 
    is_cuda = True if torch.cuda.is_available() else False
    print("CUDA Available: ", torch.cuda.is_available())
    print("Number of GPUs: ", torch.cuda.device_count())
    episode_length = 96 # actually modelled via environment
    batch_size = 48
    learning_rate = 0.0003
    input_size = 7 # cue(one_hot [4]), prev_action (one_hot [2]), prev_reward 

    if args.with_time == 'true':
        input_size += 1
        with_time = True
    else:
        with_time = False

    if args.with_counter_reward == 2:
        input_size += 1 # add dim if we concat prev_reward and counter_reward
    
    args.hidden_size = input_size * 8

    max_grad_norm = 50
    overfit = False
    value_coef = 0.5 # beta_v
    #return_coef = 0.8 # aka gamma / discount
    #entropy_coef is annealed inside the training
    #hidden_size = 48

    # model_name for tests where discount and entropy final was varied
    #model_name = f'RL__rc_{args.return_coef}__efv_{args.entropy_final_value}w__rew_{args.with_counter_reward}__id_{args.task_id}'
    # model_name for tests where hidden_size is changed
    model_name = f'{args.agent_model}__{args.train_eps}__mode_{args.env_mode}__num_layers_{args.num_layers_transformer}__hidden_size_{args.hidden_size}__with_time_{args.with_time}'
    #model_name = f'RL__train_eps_{args.train_eps}__id_{args.task_id}'
    folder_path = f'data/{args.exp_name}/{model_name}/'    

    if args.task_id == 0:
        with_random = True
    else:
        with_random = False

    os.makedirs(folder_path, exist_ok=False) 
    print(folder_path)

    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d-%H:%M:%S")

    params = {'date_time': date_time,
              'is_cuda': is_cuda,
              'task_id': args.task_id,
              'train_eps': args.train_eps,
              'trials_per_eps': episode_length,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'hidden_size': args.hidden_size,
              'input_size': input_size,
              'max_grad_norm': max_grad_norm,
              'overfit': False,
              'return_coef': args.return_coef,
              'value_coef': value_coef,
              'entropy_final_value': args.entropy_final_value,
              'loss_fn': args.loss_fn,
              'with_time': args.with_time,
              'with_random':with_random,
              'env_mode':args.env_mode,
              'with_counter_reward':args.with_counter_reward,
              'success':args.success,
              'fail':args.fail}

    with open(folder_path+"params.json", "w") as outfile:
        json.dump(params, outfile)

    writer = SummaryWriter(log_dir=folder_path+'tb/')

    a2c = A2COptim(batch_size, args.hidden_size, input_size, args.num_layers_transformer, args.n_heads,learning_rate, is_cuda, writer, agent_model=args.agent_model)
    runner = Runner(episode_length, batch_size, folder_path,a2c,writer, with_time, with_random, args.with_counter_reward)
    runner.training(args.train_eps, args.loss_fn, args.return_coef, value_coef, args.entropy_final_value, max_grad_norm, overfit, args.env_mode, args.success, args.fail)