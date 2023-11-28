"""Fit the Rescorla-Wagner models to the data."""

import sys
sys.path.append('../../rw_models')
sys.path.append('rw_models')
import pandas as pd
import numpy as np
import argparse
from scipy.optimize import minimize

from rw_agency import RW_Agency as RW

def calculate_BIC(k,n,nll):
    """
    Parameters: 
        - k: number of parameters estimated by the model
        - n: number of observations in x
        - nll: the negative log likelihood of the model
    """
    return k * np.log(n) + 2 * nll

def fitting(models, dfs, name, type):
    run = pd.DataFrame(columns = ['meta_rl_model',
                                  'agency_type',
                                    'fitting_model',
                                    'part_run',
                                    'nll',
                                    'bic',
                                    'beta',
                                    'alpha_free',
                                    'alpha_forced',
                                    'alpha_free_pos',
                                    'alpha_free_neg',
                                    'alpha_forced_pos',
                                    'alpha_forced_neg'])

    for model in models:

        meta_rl_model = name

        # create block_idx across multiple "sessions"
        df['part_run'] = ((df['run']) // 3)
        df['idx'] = (df['run'] % 3) * 2 + df['block_idx']

        print(f'Amount of participants: {df.part_run.max()+1}')
        
        for nsub in range(df.part_run.max()+1):
            print(f'Fitting {model.__name__} for participant {nsub+1}...')

            # Replace 'load' with loading the data manually, example: np.load or any other method
            M = df[df['part_run'] == nsub]

            data = np.empty((len(M), 4))

            data[:, 0] = M.actions.values + 1
            data[:, 1] = M.rewards.values
            data[:, 2] = M.idx.values + 1
            data[:, 3] = 1- M.cues.values.astype(bool).astype(int)
            data = data.astype(int)

            epsilon = 1e-10 # to avoid bounds of 0/1

            if model.__name__ == 'Model_2alpha':

                init_guess = [5] + [.5] * 2
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 2
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(3, len(data), est.fun)
                temp_run = pd.DataFrame([{'meta_rl_model': meta_rl_model,
                                          'agency_type': type,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_free': est.x[1],
                                        'alpha_forced': est.x[2]}])
                run = pd.concat([run, temp_run], axis=0)
            
            elif model.__name__ == 'Model_3alpha':

                init_guess = [5] + [.5] * 3
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 3
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(4, len(data), est.fun)
                temp_run = pd.DataFrame([{'meta_rl_model': meta_rl_model,
                                          'agency_type': type,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_free_pos': est.x[1],
                                        'alpha_free_neg': est.x[2],
                                        'alpha_forced': est.x[3]}])
                run = pd.concat([run, temp_run], axis=0)

            elif model.__name__ == 'Model_4alpha':

                init_guess = [5] + [.5] * 4
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 4
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(5, len(data), est.fun)
                temp_run = pd.DataFrame([{'meta_rl_model': meta_rl_model,
                                            'agency_type': type,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_free_pos': est.x[1],
                                        'alpha_free_neg': est.x[2],
                                        'alpha_forced_pos': est.x[3],
                                        'alpha_forced_neg': est.x[4]}])
                run = pd.concat([run, temp_run], axis=0)

    return run



# Run
parser = argparse.ArgumentParser()
parser.add_argument('--engine', type=str, default='claude-1')
parser.add_argument('--type',  type=str, default='you') # 24 (part) * 3 (sess) = 72 
args = parser.parse_args()  
engine = args.engine
type = args.type
sys.path.append('../llm_utils') # maybe not working
sys.path.append('llm/llm_utils')

# to change the models
models = [RW.Model_2alpha, RW.Model_3alpha, RW.Model_4alpha]


df = pd.read_csv(f'/u/jschubert/learning_bias/llm/agency/data/{engine}/sim_{type}.csv')

df = df[df.block_forced_type == 1.0]


run = fitting(models, df, engine, type)
run.to_csv(f'/u/jschubert/learning_bias/llm/agency/data/{engine}/rw_fit_{type}.csv', index=False)