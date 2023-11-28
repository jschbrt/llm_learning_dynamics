"""Fit the RW models to the data."""

import sys
sys.path.append('../../rw_models')
sys.path.append('rw_models')
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from rw_conf_bias import RW_ConfBias as RW

def calculate_BIC(k,n,nll):
    """
    Parameters: 
        - k: number of parameters estimated by the model
        - n: number of observations in x
        - nll: the negative log likelihood of the model
    """
    return k * np.log(n) + 2 * nll

def fitting(models, df, llm):
    run = pd.DataFrame(columns = ['llm',
                                'fitting_model',
                                'part_run',
                                'nll',
                                'bic',
                                'beta',
                                'alpha_conf',
                                'alpha_disconf',
                                'alpha_forced',
                                'alpha_free_chosen',
                                'alpha_free_unchosen',
                                'alpha_forced_chosen',
                                'alpha_forced_unchosen',
                                'alpha_free_pos_chosen',
                                'alpha_free_pos_unchosen',
                                'alpha_free_neg_chosen',
                                'alpha_free_neg_unchosen',
                                'alpha_forced_pos_chosen',
                                'alpha_forced_pos_unchosen',
                                'alpha_forced_neg_chosen',
                                'alpha_forced_neg_unchosen'])

    for model in models:
        print(f'Fitting {model.__name__}')
        # create block_idx across multiple "sessions"
        df['part_run'] = ((df['run']) // 4) # 4 sessions per participant
        df['idx'] = (df['run'] % 4) * 4 + df['block_idx']

        print(f'Amount of data: {df.part_run.max()+1}')

        for nsub in range(df.part_run.max()+1):
            print(f'{nsub+1}')

            # Replace 'load' with loading the data manually, example: np.load or any other method
            M = df[df['part_run'] == nsub]

            data = np.empty((len(M), 7))

            data[:, 0] = M.actions.values + 1
            data[:, 1] = M.rewards.values
            data[:, 2] = M.idx.values + 1
            data[:, 3] = 1- M.cues.values.astype(bool).astype(int)
            data[:, 4] = M.counter_actions.values + 1
            data[:, 5] = M.forgone_rewards.values
            data[:, 6] = M.block_feedback_type.values

            data = data.astype(int)

            epsilon = 1e-10 # to avoid bounds of 0/1

            if model.__name__ == 'Model_3alpha':

                init_guess = [5] + [.5] * 3
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 3
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(4, len(data), est.fun)
                temp_run = pd.DataFrame([{'llm': llm,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_conf': est.x[1],
                                        'alpha_disconf': est.x[2],
                                        'alpha_forced': est.x[3]}])
                run = pd.concat([run, temp_run], axis=0)
            
            elif model.__name__ == 'Model_4alpha':

                init_guess = [5] + [.5] * 4
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 4
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(5, len(data), est.fun)
                temp_run = pd.DataFrame([{'llm': llm,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_free_chosen': est.x[1],
                                        'alpha_free_unchosen': est.x[2],
                                        'alpha_forced_chosen': est.x[3],
                                        'alpha_forced_unchosen': est.x[4]}])
                run = pd.concat([run, temp_run], axis=0)

            elif model.__name__ == 'Model_6alpha':

                init_guess = [5] + [.5] * 6
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 6
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(7, len(data), est.fun)
                temp_run = pd.DataFrame([{'llm': llm,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_free_pos_chosen': est.x[1],
                                        'alpha_free_neg_chosen': est.x[2],
                                        'alpha_free_pos_unchosen': est.x[3],
                                        'alpha_free_neg_unchosen': est.x[4],
                                        'alpha_forced_chosen': est.x[5],
                                        'alpha_forced_unchosen': est.x[6]}])
                run = pd.concat([run, temp_run], axis=0)

            elif model.__name__ == 'Model_8alpha':

                init_guess = [5] + [.5] * 8
                bounds = [(0+epsilon, np.inf)] + [(0+epsilon,1-epsilon)] * 8
                est = minimize(model, init_guess, data, bounds=bounds)

                bic = calculate_BIC(9, len(data), est.fun)
                temp_run = pd.DataFrame([{'llm': llm,
                                        'fitting_model': model.__name__,
                                        'part_run': nsub,
                                        'nll': est.fun,
                                        'bic': bic,
                                        'beta': est.x[0],
                                        'alpha_free_pos_chosen': est.x[1],
                                        'alpha_free_neg_chosen': est.x[2],
                                        'alpha_free_pos_unchosen': est.x[3],
                                        'alpha_free_neg_unchosen': est.x[4],
                                        'alpha_forced_pos_chosen': est.x[5],
                                        'alpha_forced_neg_chosen': est.x[6],
                                        'alpha_forced_pos_unchosen': est.x[7],
                                        'alpha_forced_neg_unchosen': est.x[8]}])
                run = pd.concat([run, temp_run], axis=0)

    return run


# to change the models
name = 'claude-1'

models = [RW.Model_3alpha, RW.Model_4alpha, RW.Model_6alpha, RW.Model_8alpha]

file = f'/u/jschubert/llm_experiments/conf_bias/data/{name}/sim.csv'
df = pd.read_csv(file)
df['counter_actions'] = 1.0 - df['actions']


run = fitting(models, df, name)
run.to_csv(f'/u/jschubert/llm_experiments/conf_bias/data/{name}/rw_fit.csv', index=False)