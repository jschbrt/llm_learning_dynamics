import sys
sys.path.append('../../rw_models')
sys.path.append('rw_models')

import pandas as pd
import numpy as np

from scipy.stats import gamma, beta
from scipy.optimize import minimize
import scipy.io

from rw_conf_bias import RW_ConfBias as RW

def calculate_BIC(k,n,nll):
    """
    Parameters: 
        - k: number of parameters estimated by the model
        - n: number of observations in x
        - nll: the negative log likelihood of the model
    """
    return k * np.log(n) + 2 * nll


models = [RW.Model_3alpha, RW.Model_4alpha, RW.Model_6alpha, RW.Model_8alpha]

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
    print(model.__name__)

    llm = 'human'

    for nsub in range(1,25):
        print(nsub)

        path = '/u/jschubert/optimism_transformer/palminteri/original/Experiment2/'

        # Replace 'load' with loading the data manually, example: np.load or any other method
        data = scipy.io.loadmat(path + 'passymetrieII_Suj' + str(nsub) + '.mat')
        M = data['M']

        data = np.empty((len(M), 7))

        for i in range(len(M)):
            data[i, 0] = M[i, 5] + 1 # participant choice (+1 -> 1 = left, 2 = right)
            data[i, 1] = M[i, 3] * (M[i, 5] == 1) + M[i, 4] * (M[i, 5] == 0) # outcome for chosen option
                        # 3 -> best_rewarded / 4 -> worst rewarded / 5 -> participant choice (0 (worst), 1 (best))
            data[i, 2] = M[i, 1] + 4 * (M[i, 0] - 1)  # block number (0-15)
            data[i, 3] = M[i, 6] # free (1) forced (0)
            data[i, 4] = (1 - M[i, 5]) + 1
            data[i, 5] = M[i, 3] * ((data[i,4]-1) == 1) + M[i, 4] * ((data[i,4]-1) == 0) # outcome for unchosen option
                        # 3 -> best_rewarded / 4 -> worst rewarded / 5 -> participant choice (0 (worst), 1 (best))
        
        data[:, 6] = ([0]*40 + [1]*40)*8 # partial, full, partial,

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

run.to_csv('/u/jschubert/optimism_transformer/palminteri/rw_fitting_exp2.csv', index=False)