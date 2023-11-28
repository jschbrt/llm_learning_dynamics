import sys
sys.path.append('../../rw_models')
sys.path.append('rw_models')

import pandas as pd
import numpy as np

from scipy.stats import gamma, beta
from scipy.optimize import minimize

from rw_agency import RW_Agency as RW

# For all three models

def calculate_BIC(k,n,nll):
    """
    Parameters: 
        - k: number of parameters estimated by the model
        - n: number of observations in x
        - nll: the negative log likelihood of the model
    """
    return k * np.log(n) + 2 * nll


files = ['/u/jschubert/learning_bias/meta-rl/agency/data/agency_blockwise_trained/gpu_eps10000_bs64_agency_test_mask_policy_value_loss_20231008_13:00/test/simulation_df.csv']

names = ['mask_policy_value_loss']

dfs = [pd.read_csv(file) for file in files]

models = [RW.Model_2alpha, RW.Model_3alpha, RW.Model_4alpha]

run = pd.DataFrame(columns = ['meta_rl_model',
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
                              'alpha_forced_neg',])

for model in models:

    for i, df in enumerate(dfs):

        meta_rl_model = names[i]

        max_blocks = df.batch_idx.max()+1

        df['single_run'] = df['test_eps_idx'] * max_blocks + df['batch_idx'] # 48*2=96 runs

        # create block_idx across multiple "sessions"
        df['part_run'] = ((df['single_run']) // 3)
        df['idx'] = (df['single_run'] % 3) * 4 + df['block_idx']

        for nsub in range(24):

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

run.to_csv('/u/jschubert/learning_bias/meta-rl/agency/data/agency_blockwise_trained/rw_fitting.csv', index=False)