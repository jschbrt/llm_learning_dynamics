import sys
sys.path.append('../../rw_models')
sys.path.append('rw_models')

import pandas as pd
import numpy as np

from scipy.stats import gamma, beta
from scipy.optimize import minimize
import scipy.io

from rw_agency import RW_Agency as RW

def calculate_BIC(k,n,nll):
    """
    Parameters: 
        - k: number of parameters estimated by the model
        - n: number of observations in x
        - nll: the negative log likelihood of the model
    """
    return k * np.log(n) + 2 * nll


file = '/u/jschubert/optimism_transformer/palminteri/simulation_humans.csv'
name = 'humans'

df = pd.read_csv(file)
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

    meta_rl_model = name

    for nsub in range(1,25):

        path = '/u/jschubert/optimism_transformer/palminteri/original/Experiment1/'

        # Replace 'load' with loading the data manually, example: np.load or any other method
        data = scipy.io.loadmat(path + 'passymetrieI_Suj' + str(nsub) + '.mat')
        M = data['M']

        data = np.empty((len(M), 4))


        for i in range(len(M)):
            data[i, 0] = M[i, 5] + 1 # participant choice (+1 -> 1 = left, 2 = right)
            data[i, 1] = M[i, 3] * (M[i, 5] == 1) + M[i, 4] * (M[i, 5] == 0) # outcome for chosen option
                        # 3 -> best_rewarded / 4 -> worst rewarded / 5 -> participant choice (0 (worst), 1 (best))
            data[i, 2] = M[i, 1] + 4 * (M[i, 0] - 1)  # block number (0-15)
            data[i, 3] = M[i, 6] # free (1) forced (0)

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

run.to_csv('/u/jschubert/optimism_transformer/palminteri/rw_fitting_3models.csv', index=False)