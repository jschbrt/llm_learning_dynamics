import numpy as np
from scipy.stats import gamma, beta

class RW_Agency:
    '''
    Contains three fitting models with different number of parameters.
    These models are used to fit the data from the agency task.
    '''

    def Model_2alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:3], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                lik += params[0] * Q[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q[data[i, 2] - 1, :])))
                # update q value of chosen option
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI
            else:
                # update the same q values but with different alphas
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[2] * deltaI

        return -(priors + lik)

    def Model_3alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:4], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                lik += params[0] * Q[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI * (deltaI > 0) + params[2] * deltaI * (deltaI < 0)
            else:
                # update the same q values but with different alphas
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI

        return -(priors + lik)

    def Model_4alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:5], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                lik += params[0] * Q[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI * (deltaI > 0) + params[2] * deltaI * (deltaI < 0)
            else:
                # update the same q values but with different alphas
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI * (deltaI > 0) + params[4] * deltaI * (deltaI < 0)
        return -(priors + lik)

class RW_No_Forced:
    '''
    Contains three fitting models with different number of parameters.
    These models are used to fit the data from the agency task but without forced choices.
    '''

    def Model_1alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                lik += params[0] * Q[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q[data[i, 2] - 1, :])))
                # update q value of chosen option
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI
            else:
                # update the same q values but with different alphas
                # Q[data[i, 2] - 1, data[i, 0] - 1] += params[2] * deltaI
                pass

        return -(priors + lik)

    def Model_2alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:3], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI = data[i, 1] - Q[data[i, 2]-1, data[i, 0]-1] # 

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                lik += params[0] * Q[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI * (deltaI > 0) + params[2] * deltaI * (deltaI < 0)
            else:
                # update the same q values but with different alphas
                # Q[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI * (deltaI > 0) + params[4] * deltaI * (deltaI < 0)
                pass
        return -(priors + lik)

