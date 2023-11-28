import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.optimize import minimize, differential_evolution
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
import os

def rescorla_wagner(params, choices, outcomes, plus_minus):

    if plus_minus:
        alpha_plus, alpha_minus, inverse_temperature = params
    else:
        learning_rate, inverse_temperature = params
    
    # Initialize values
    values = np.ones(2) * 0.25 #TODO: The 0.25 instead of 0.5 is to see if it matches the paper which was 0 or 0.5 dollars
    outcomes *= 0.5 #TODO: This is to see if it matches the paper which was 0 or 0.5 dollars
    # values = np.ones(2) * 0
    
    # Log-likelihood
    log_likelihood = 0
    for choice, outcome in zip(choices, outcomes):

        # Softmax computation with normalization
        logits = torch.tensor([inverse_temperature * values[0], inverse_temperature * values[1]])
        logits = logits - torch.max(logits)  # Subtract the maximum value for numerical stability
        probabilities = F.softmax(logits, dim=0)
        likelihood = probabilities[choice]
        epsilon = 1e-4
        likelihood = np.clip(likelihood, epsilon, 1-epsilon)
        log_likelihood += torch.log(likelihood)

        # Update values
        prediction_error = outcome - values[choice]
        if plus_minus:
            if prediction_error > 0:
                values[choice] +=  alpha_plus * prediction_error
            else:
                values[choice] +=  alpha_minus * prediction_error
        else: 
            values[choice] += learning_rate * prediction_error
        


    return log_likelihood

def rescorla_wagner_across_runs( params, path, runs, plus_minus):
    # Get the log likelihoods for all runs
    # runs = len(glob.glob(f'{path}/run_*.csv'))
    log_likelihood = np.zeros((runs[1] - runs[0]))  #TODO: CHECK THIS BS
    for count, run in enumerate(range(runs[0], runs[1])):
        df = pd.read_csv(f'{path}/run_' + str(run) + '.csv')
        for casino in range(1, 5):
            df_casino = df[df['casino'] == casino]
            choices = df_casino['choice']
            outcomes = df_casino['rewards']
            log_likelihood[count] += rescorla_wagner(params, choices, outcomes, plus_minus)

    return -np.mean(log_likelihood)

def fit_rescorla_wagner(path, runs, plus_minus=False):
    # Initial parameter values
    bounds = [(0, 1), (0, 1e5)] if not plus_minus else [(0, 1), (0, 1), (0, 1e5)]

    # Maximum likelihood estimation
    # best_nll = np.inf
    # for alpha_guess in tqdm(np.linspace(0, 1, 10)):
    #     for beta_guess in np.linspace(0, 20, 5):
    #         initial_params = [alpha_guess, beta_guess] if not plus_minus else [alpha_guess, alpha_guess, beta_guess]

    #         result = minimize(rescorla_wagner_across_runs, initial_params, args=(path, plus_minus) ,method='L-BFGS-B', bounds=bounds)
    #         if result.fun < best_nll:
    #             best_nll = result.fun
    #             best_params = result.x

    #! Use differential evolution
    result = differential_evolution(rescorla_wagner_across_runs,bounds=bounds, args=(path, runs, plus_minus), maxiter=100)
    best_nll = result.fun
    best_params = result.x
    #!
    # Return the best parameters and the negative log-likelihood
    return best_params, best_nll



# Run
parser = argparse.ArgumentParser()
parser.add_argument('--engines', nargs='+', default=['all'])
parser.add_argument('--no_participants',  default=10)
args = parser.parse_args()
engines = args.engines
no_participants = args.no_participants

# if engines == ['all']: get all engines which are subfolders in the data folder
if engines == ['all']:
    engines = [file for file in os.listdir('./data') if os.path.isdir(f'./data/{file}') if file != 'humans'] #TODO: CHECK what to do with humans still

    # df = pd.read_csv('../Benchmark/final_data.csv')
    # engines = [engine for engine in engines if df[(df['score type'] == 'Learning rate') & (df['Engine'] == engine)].participant_no.max() <= no_participants]

for engine in tqdm(engines):
    path = f'data/{engine}'
    # no_participants =  len(glob.glob(f'{path}/run_*.csv'))  #Because each run is a participant
    for participant in range(no_participants):
        print(f'For engine: {engine}-------------------------------------------')
        runs = (participant, participant+1)
        params_rw, nll_rw = fit_rescorla_wagner(path, runs)
        params_rwpm, nll_rwpm = fit_rescorla_wagner(path, runs, plus_minus=True)
        # Get the BICs for normal rw
        # n_trials = 96 * len(glob.glob(f'{path}/run_*.csv'))
        n_trials = 96 

        # Get the BIC for plus minus rw
        BIC_rw = 2*nll_rw + len(params_rw) * np.log(n_trials)
        print(f'For engine: {engine}-------------------------------------------')
        print(f'Best parameters: {params_rw}')
        print(f'Negative log-likelihood: {nll_rw}')
        print(f'BIC for normal rw: {BIC_rw}')

        BIC_rwpm = 2*nll_rwpm + len(params_rwpm) * np.log(n_trials)
        print(f'Best parameters: {params_rwpm}')
        print(f'Negative log-likelihood: {nll_rwpm}')
        print(f'BIC for pm rw: {BIC_rwpm}')

        # Add the final score to the csv file
        lr = params_rw[0] #0 to 1
        optimismbias = params_rwpm[0] - params_rwpm[1] #-1 to 1
        file_path = f'../Benchmark/data/{engine}.csv'
        dict1 = {'score type': 'Learning rate', 'score': lr, 'extra notes': 'space is 0 to 1'}
        dict2 = {'score type': 'Optimism Bias', 'score': optimismbias, 'extra notes': f'alpha+:{params_rwpm[0]} - alpha-:{params_rwpm[1]}& space is -1 to 1'}
        dict = [dict1, dict2]
        if no_participants == 1:
            for i in range(2):
                # store the score in the csv file
                df = pd.read_csv(file_path)
                # check if the score type is already in the csv
                if dict[i]['score type'] not in df['score type'].values:
                    df = df.append(dict[i], ignore_index=True)
                else:
                    df.loc[df['score type'] == dict[i]['score type'], 'score'] = dict[i]['score']
                df.to_csv(file_path, index=False)
        else:
            score_types = ['Learning rate', 'Optimism Bias']
            scores = [lr, optimismbias]
            # Add the final score to the final csv where each row is an engine row seen as a participant
            storing_df = pd.read_csv('../Benchmark/final_data.csv')
            llm_info_df = pd.read_csv('../Benchmark/llm_features.csv')
            #If already exists in storing_df then replace it don't add it
            # So check if there is a row with the same engine, score type and participant_no
            for idx, score_type in enumerate(score_types):
                if len(storing_df[(storing_df['Engine'] == engine) & (storing_df['score type'] == score_type) & (storing_df['participant_no'] == participant)]) > 0:
                    storing_df.loc[(storing_df['Engine'] == engine) & (storing_df['score type'] == score_type) & (storing_df['participant_no'] == participant), 'score'] = scores[idx]
                else:
                    new_row = llm_info_df[llm_info_df['Engine'] == engine].iloc[0].to_dict()
                    new_row['score type'] =  score_type
                    new_row['score'] = scores[idx]
                    new_row['participant_no'] = participant
                    storing_df = pd.concat([storing_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                storing_df.to_csv('../Benchmark/final_data.csv', index=False) 




        