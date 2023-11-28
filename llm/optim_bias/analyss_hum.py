import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.optimize import minimize, differential_evolution
from tqdm import tqdm
import torch
import torch.nn.functional as F

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

        # Calculate likelihood of the choice
        # likelihood = np.exp(inverse_temperature * values[choice]) /  (np.exp(inverse_temperature * values[0]) + np.exp(inverse_temperature * values[1])) #TODO ensure there s no overflow with high Betas
        # #! Avoid overflow
        # epsilon = 1e-10
        # likelihood = np.clip(likelihood, epsilon, 1-epsilon)
        # log_likelihood += np.log(likelihood)

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


def rescorla_wagner_across_runs( params, path, plus_minus, run):
    # Get the log likelihoods for all runs
    # runs = len(glob.glob(f'{path}/run_*.csv'))
    # log_likelihood = np.zeros(runs)
    # for run in range(runs):
    #     df = pd.read_csv(f'{path}/run_' + str(run) + '.csv')
        # for casino in range(1, 5):
        #     df_casino = df[df['casino'] == casino]
        #     choices = df_casino['choice']
        #     outcomes = df_casino['rewards']
        #     log_likelihood[run] += rescorla_wagner(params, choices, outcomes, plus_minus)


    #! Human data
    df = pd.read_csv(f'{path}/run_' + str(run) + '.csv')
    log_likelihood = 0
    for casino in range(1, 5):
        df_casino = df[df['casino'] == casino]
        choices = df_casino['choice']
        outcomes = df_casino['rewards']
        log_likelihood += rescorla_wagner(params, choices, outcomes, plus_minus)

    # return -np.mean(log_likelihood)
    return -log_likelihood

def fit_rescorla_wagner(path, plus_minus=False):
    # Initial parameter values
    bounds = [(0, 1), (0, 1e6)] if not plus_minus else [(0, 1), (0, 1), (0, 1e6)]

    # Maximum likelihood estimation
    # best_nll = np.inf
    # for alpha_guess in tqdm(np.linspace(0, 1, 10)):
    #     for beta_guess in np.linspace(0, 10, 5):
    #         initial_params = [alpha_guess, beta_guess] if not plus_minus else [alpha_guess, alpha_guess, beta_guess]

    #         result = minimize(rescorla_wagner_across_runs, initial_params, args=(path, plus_minus) ,method='L-BFGS-B', bounds=bounds)
    #         if result.fun < best_nll:
    #             best_nll = result.fun
    #             best_params = result.x

    #! Use differential evolution
    runs = len(glob.glob(f'{path}/run_*.csv'))
    best_nlls = np.zeros(runs)
    best_params = np.zeros((runs, len(bounds)))
    for run in tqdm(range(runs)):
        result = differential_evolution(rescorla_wagner_across_runs, bounds, args=(path, plus_minus, run), maxiter=100)
        best_nlls[run] = result.fun
        best_params[run] = result.x
        #!
    # Return the best parameters and the negative log-likelihood
    return np.mean(best_params, axis=0), np.mean(best_nlls)

if __name__ == "__main__":
    # engines = [ 'llama_65', 'humans']
    # engines = ['llama_65']
    engines = ['humans']
    for engine in engines:
        path = f'data/{engine}'
        params_rw, nll_rw = fit_rescorla_wagner(path)
        params_rwpm, nll_rwpm = fit_rescorla_wagner(path, plus_minus=True)
        # Get the BICs for normal rw
        n_trials = 96 * len(glob.glob(f'{path}/run_*.csv'))

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


        # Try paper parameter values
        # alpha = 0.32
        # beta = 1/0.16
        # ngl = rescorla_wagner_across_runs( [alpha, beta], path, plus_minus=False)
        # print(f'Negative log-likelihood: {ngl}')
        # print(f'BIC for normal rw: {2*ngl +  2*np.log(96)}')

        # alpha1 = 0.36
        # alpha2 = 0.22
        # beta = 1/0.13

        # ngl = rescorla_wagner_across_runs( [alpha1, alpha2, beta], path, plus_minus=True)
        # print(f'Negative log-likelihood: {ngl}')
        # print(f'BIC for pm rw: {2*ngl +  3*np.log(96)}')        