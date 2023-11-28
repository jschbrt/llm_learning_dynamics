# %%
import numpy as np
from scipy import optimize as opt
import torch.nn.functional as F
import torch
from numpy.random import rand
#from tqdm import tqdm
import pandas as pd

# Fixed the bug that assigning a list of self to var with = will just point to old list and changes it as well

class predict_exp():
    """
    This class is used for fitting two models for the optimism bias task.
    """

    def __init__(self, alpha_range, beta_range, q_initial, n_optim_inits, optim_bounds, concatenated_trial=False) -> None:
        """
        Parameters:
            - q_initial: initial Q values for the RW model, changes per experiment
            - alpha_range: min, max for optimization procedure
            - beta_range: min, max for optimization procedure
        """
        self.r_a_min, self.r_a_max = alpha_range
        self.r_b_min, self.r_b_max = beta_range
        self.q_initial = q_initial
        self.n_inits = n_optim_inits

        a_bound, b_bound = optim_bounds
        self.c_optim_bounds = (a_bound, b_bound)
        self.pm_optim_bounds = (a_bound, a_bound, b_bound)
        self.concatenated_trial = concatenated_trial

    def fit(self, choices, rewards, contexts, bounded=False):
        """
        This models the estimate of the expected values for action 1 and action 2 on the basis of individual sequences of choices of individual outcomes.
        Parameters:
            - c: list of choices / actions taken at time t
            - r: list of rewards received while performing action at t
        """
        res_participants = []
        for p in range(choices.shape[0]):

            c = choices[p,:]
            r = rewards[p,:]
            context = contexts[p,:]

            c_nll, c_a, c_b = self.c_fit(c, r, context, bounded)
            pm_nll, p_a, m_a, pm_b = self.pm_fit(c, r, context, bounded)
            
            c_bic = self.calculate_BIC(2, len(c), c_nll)
            pm_bic = self.calculate_BIC(3, len(c), pm_nll)
    
            res_participants.append((p, c_bic, pm_bic, c_nll, pm_nll, c_a, p_a, m_a, c_b, pm_b))

        df = pd.DataFrame(res_participants, columns=['participant_idx', 'c_bic', 'pm_bic', 'c_nll', 'pm_nll', 'c_a', 'p_a', 'm_a', 'c_b', 'pm_b'])
        return df
    
    def c_fit(self, c, r, context, bounds=False):
        nll = np.inf
        pred_a = np.nan
        pred_b = np.nan
        for _ in range(self.n_inits):
            init_guess = self._generate_sample()
            if bounds:
                est = opt.minimize(self.c_nll, 
                            init_guess,
                            (c,r, context),
                            bounds=self.c_optim_bounds)
            else:
                est = opt.minimize(self.c_nll, 
                                init_guess,
                                (c,r, context))
            if est.fun < nll:
                nll = est.fun
                pred_a = est.x[0]
                pred_b = est.x[1]
        return nll, pred_a, pred_b

    def c_nll(self, params, c, r, context):
        alpha, beta = params
        q = self.q_initial.copy()# to avoid that it is only assigning the same object to a different name
        q_list = np.zeros((len(c), 2))
        nll_loop = 0
        
        for t in range(len(c)):

            if self.concatenated_trial:
                if t % 96 == 0:
                    q = self.q_initial.copy()

            q_list[t,:] = q[context[t]]

            log_p_loop = self._log_softmax(q[context[t],:],beta,in_loop=True)
            nll_loop += log_p_loop[c[t]].item()

            q = self._c_rw_update(q,c,r, context,t,alpha)
        
        log_p = self._log_softmax(q_list, beta)

        nll = F.nll_loss(log_p, torch.tensor(c), reduction='sum') # basically it sums log_p[c[t]] and then divides it by len(c) and adds a -
        if not np.isclose(-nll_loop,nll.item()):
            print('not close')
        return nll.item()

    def _c_rw_update(self, q, c, r, context, t, alpha):
        delta = r[t] - q[context[t],c[t]]
        q[context[t],c[t]] = q[context[t],c[t]] + alpha * delta
        return q

    def pm_fit(self, c, r, context, bounds=False):
        nll = np.inf
        pred_p_a = np.nan
        pred_m_a = np.nan
        pred_b = np.nan
        for _ in range(self.n_inits):
            init_guess = self._generate_sample(pm=True)
            if bounds:
                est = opt.minimize(self.pm_nll, 
                               init_guess,
                               (c,r, context),
                               bounds=self.pm_optim_bounds)
            else:
                est = opt.minimize(self.pm_nll, 
                               init_guess,
                               (c,r, context))
            if est.fun < nll:
                nll = est.fun
                pred_p_a = est.x[0]
                pred_m_a = est.x[1]
                pred_b = est.x[2]
        return nll, pred_p_a, pred_m_a, pred_b

    def pm_nll(self, params, c,r, context):
        p_a, m_a, beta = params
        q = self.q_initial.copy() # to avoid that it is only assigning the same object to a different name
        q_list = np.zeros((len(c), 2))
        
        for t in range(len(c)):
            q_list[t,:] = q[context[t]]
            q = self._pm_rw_update(q,c,r,context,t,p_a,m_a)
        
        log_p = self._log_softmax(q_list, beta)
        nll = F.nll_loss(log_p, torch.tensor(c), reduction='sum') # basically it sums log_p[c[t]] and then divides it by len(c) and adds a -
        return nll.item()
    
    def _pm_rw_update(self, q, c, r, context, t, p_a, m_a):
        delta = r[t] - q[context[t],c[t]]
        if delta > 0:
            q[context[t],c[t]] = q[context[t],c[t]] + p_a * delta
        else:
            q[context[t],c[t]] = q[context[t],c[t]] + m_a * delta
        return q

    def _log_softmax(self,q_values, beta, in_loop=False):
        q_values = np.multiply(q_values, beta)
        if in_loop:
            log_p = F.log_softmax(torch.tensor(q_values), dim=0) # this yielded the same result as my own implementation
        else:
            log_p = F.log_softmax(torch.tensor(q_values), dim=1)
        return log_p

    def calculate_BIC(self, k,n,nll):
        """
        Parameters: 
            - k: number of parameters estimated by the model
            - n: number of observations in x
            - nll: the negative log likelihood of the model
        """
        return k * np.log(n) + 2 * nll

    def _generate_sample(self, pm=False):
        if pm:
            r_a1 = (self.r_a_min + rand(1) * (self.r_a_max - self.r_a_min))[0]
            r_a2 = (self.r_a_min + rand(1) * (self.r_a_max - self.r_a_min))[0]
            r_b = (self.r_b_min + rand(1) * (self.r_b_max - self.r_b_min))[0]
            return [r_a1, r_a2, r_b]
        else:
            r_a = (self.r_a_min + rand(1) * (self.r_a_max - self.r_a_min))[0]
            r_b = (self.r_b_min + rand(1) * (self.r_b_max - self.r_b_min))[0]
            return [r_a, r_b]