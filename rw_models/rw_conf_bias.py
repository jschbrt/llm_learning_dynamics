import numpy as np
from scipy.stats import gamma, beta

class RW_ConfBias:
    """
    Contains four fitting models with different number of parameters.
    These models are used to fit the data from the confirmation bias task.
    """

    def Model_3alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:4], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0

        Q_ = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 
            deltaI_u = data[i, 5] - Q_[data[i, 2]-1, data[i, 4]-1]

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
                if data[i, 6] == 1:
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[2] * deltaI_u * (deltaI_u > 0) + params[1] * deltaI_u * (deltaI_u < 0)
            else:
                # update the same q values but with different alphas
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI_c
                if data[i, 6] == 1:
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[3] * deltaI_u

        return -(priors + lik)

    def Model_4alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:5], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q_ = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 
            deltaI_u = data[i, 5] - Q_[data[i, 2]-1, data[i, 4]-1]

            if data[i, 3] == 1: # if free choice
                lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c
                if data[i, 6] == 1:
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[2] * deltaI_u
            else:
                # update the same q values but with different alphas
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[3] * deltaI_c
                if data[i, 6] == 1:
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[4] * deltaI_u

        return -(priors + lik)

    def Model_6alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:7], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q_ = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 
            deltaI_u = data[i, 5] - Q_[data[i, 2]-1, data[i, 4]-1]

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
                if data[i, 6] == 1:
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[3] * deltaI_u * (deltaI_u > 0) + params[4] * deltaI_u * (deltaI_u < 0)
            else:
                # update the same q values but with different alphas
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[5] * deltaI_c
                if data[i, 6] == 1:
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[6] * deltaI_u
    
        return -(priors + lik)

    def Model_8alpha(params, data):
        prior_beta = np.log(gamma.pdf(params[0], 1.2, loc=0, scale=5.0))
        prior_alphas = np.log(beta.pdf(params[1:9], 1.1, 1.1))
        priors = prior_beta + np.sum(prior_alphas)
        lik = 0
        
        Q_ = np.zeros((int(np.max(data[:, 2])), int(np.max(data[:, 0])))) # Q-values for each option in each block

        for i in range(len(data)):
            # prediction error = outcome - q value of block and action
            deltaI_c = data[i, 1] - Q_[data[i, 2]-1, data[i, 0]-1] # 
            deltaI_u = data[i, 5] - Q_[data[i, 2]-1, data[i, 4]-1]

            if data[i, 3] == 1: # if free choice
                # likelihood += beta * Q[block, option] - np.log(np.sum(np.exp(beta * Q[block, all options])))
                lik += params[0] * Q_[data[i, 2] - 1, data[i, 0] - 1] - np.log(np.sum(np.exp(params[0] * Q_[data[i, 2] - 1, :])))
                # update q value of chosen option
                # Q[block, option] += alpha_pos * deltaI * if(deltaI > 0) + alpha_neg * deltaI * if(deltaI < 0)
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[1] * deltaI_c * (deltaI_c > 0) + params[2] * deltaI_c * (deltaI_c < 0)
                if data[i, 6] == 1: # full feedback
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[3] * deltaI_u * (deltaI_u > 0) + params[4] * deltaI_u * (deltaI_u < 0)
            else:
                # update the same q values but with different alphas
                Q_[data[i, 2] - 1, data[i, 0] - 1] += params[5] * deltaI_c * (deltaI_c > 0) + params[6] * deltaI_c * (deltaI_c < 0)
                if data[i, 6] == 1: # full feedback
                    Q_[data[i, 2] - 1, data[i, 4] - 1] += params[7] * deltaI_u * (deltaI_u > 0) + params[8] * deltaI_u * (deltaI_u < 0)
        return -(priors + lik)
