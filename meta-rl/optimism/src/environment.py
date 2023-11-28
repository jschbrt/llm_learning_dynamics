"""
Contains a simple Bandit Class for Bandits with dependent or independent arms.
"""

import numpy as np
import torch
import pandas as pd

class TwoArmedBandit():
    """A stationary two-armed Bernoulli bandit."""

    def __init__(self, distribution:str, batch_size:int, success_reward=1, fail_reward=0) -> None:
        """A stationary two-armed Bernoulli bandit
        
        Args:
            :distribution: A string that specifies the sample distribution
            :success_reward: The reward on success (default: 1.)
            :fail_reward: The reward on failure (default: 0.)
        """
        dist = distribution

        if dist == 'independent': 
            self.arms_p = np.random.uniform(size=(batch_size,2))
        elif dist == 'dependent':
            first_prob = np.random.uniform(size=batch_size)
        elif dist == 'easy':
            first_prob = np.random.choice([0.1,0.9], size=batch_size)
        elif dist == 'medium':
            first_prob = np.random.choice([0.25,0.75], size=batch_size)
        elif dist == 'hard':
            first_prob = np.random.choice([0.4,0.6], size=batch_size)
        elif dist == 'overfit': # just for testing purposes
            first_prob = np.array([0.1] * batch_size)
        else:
            raise ValueError('Distribution has to be either "independent", "dependent", "easy", "medium" or "hard".')

        if dist not in ['independent']:
            self.arms_p = list(zip(first_prob, 1 - first_prob)) #type:ignore
        
        self._number_of_arms = len(self.arms_p[0]) # 2
        self._s = success_reward
        self._f = fail_reward

        self.arms_p = np.array(self.arms_p).round(2)

        # calculates the expected reward per arm and saves it as an np.array
        self.expected_rewards = self.arms_p * self._s + (1 - self.arms_p) * self._f 

    def sample(self, actions):
        """The step function.
        
        Args:
            action: An integer that specifies which arm to pull
            size: how often this action should be done

        Returns:
            A reward sampled according to the success probability of the selected arm. 

        Raises:
            ValueError: when the provided action is out of bound. 
        """
        if np.any(np.less(actions,0)) or np.any(np.greater_equal(actions, self._number_of_arms)):
            raise ValueError('An action is out of bounds fo a '
                             '{}-armed bandit'.format(self._number_of_arms))

        rewards = self.rewards(actions)
        regrets = self.regrets(actions)
        optimal_action = self.optimal_actions()

        return torch.from_numpy(rewards), torch.from_numpy(regrets), optimal_action

    def rewards(self, actions):
        probs = self.arms_p[np.arange(len(actions)), actions] # type:ignore # in array, select element [row,column]
        successes = np.random.binomial(1, probs) # for each prob, draw a sample between 0,1 for success
        rewards = successes * self._s + np.logical_not(successes) * self._f
        return rewards
        
    def regrets(self, actions):
        """Computes the regret for the given action."""
        return self.max_expected_reward() - self.expected_rewards[np.arange(len(actions)), actions] # per row, select column

    def max_expected_reward(self):
        """Returns the max value for the bandit."""
        return self.expected_rewards.max(axis=1)

    def optimal_actions(self):
        """Returns the optimal action"""
        return np.argmax(self.expected_rewards, axis=1)
    
    def suboptimal_pulls(self, actions):
        """Returns TRUE if taken action was suboptimal."""
        isSuboptimal = True
        suboptimal_pulls = len(np.where(actions != self.optimal_actions())[0])
        suboptimal_pulls = suboptimal_pulls / len(actions)
        if suboptimal_pulls == 0: # 
            isSuboptimal = False
        return suboptimal_pulls, isSuboptimal

"""
dists = ["dependent", "independent", "easy", "hard", "medium", "overfit"]
bandit = TwoArmedBandit('easy', 1)
print(bandit.arms_p)
print(bandit.sample_rewards_and_regrets([1]))
"""

class OptimBiasTask():
    """
    Optimism Bias Task containing four two-armed bandits with stationary 
    or dynamic probabilities depending on wether the mode is set to 'train' or 'test'.
    """

    def __init__(self, batch_size, with_random, mode='', success=1., fail=-1., p_idx=-1) -> None:
        self.number_cues = 4
        self.trials_per_cue = 24
        self.batch_size = batch_size
        self.mode = mode 
        self.with_random = with_random
        self.success = success
        self.fail = fail

        if self.mode.startswith('train'):
            # use an independent sample for training          
            self.prob_list_cues = np.random.uniform(size=(batch_size,8))
            self.prob_list_cues = np.reshape(self.prob_list_cues, (batch_size,4,2))
            self.prob_list_cues = self.prob_list_cues.round(2)


        if self.mode.endswith('_0.5_0'):
            self.success = 0.5
            self.fail = 0.
        elif self.mode.endswith('_0.5_-0.5'):
            self.success = 0.5
            self.fail = -0.5
        elif self.mode.endswith('_2_0'):
            self.success = 2.
            self.fail = 0.
        elif self.mode.endswith('_10_0'):
            self.success = 10.
            self.fail = 0.

        if self.mode.startswith('low_prob'):
            # use a static cue list 
            self.prob_list_cues = [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.4, 0.1],
                [0.3, 0.2]
            ]
            self.prob_list_cues = np.repeat(np.expand_dims(self.prob_list_cues,0),batch_size, axis=0)

        elif self.mode.startswith('high_prob'):
            # use a static cue list 
            self.prob_list_cues = [
                [0.9, 0.8],
                [0.8, 0.9],
                [0.6, 0.9],
                [0.7, 0.8]
            ]
            self.prob_list_cues = np.repeat(np.expand_dims(self.prob_list_cues,0),batch_size, axis=0)


        if self.mode.startswith('test'):
            # use a static cue list 
            self.prob_list_cues = [
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75]
            ]
            self.prob_list_cues = np.repeat(np.expand_dims(self.prob_list_cues,0),batch_size, axis=0)
        if self.mode == 'test_2':
            # use a static cue list 
            self.prob_list_cues = [
                [0.25, 0.75],
                [0.25, 0.75],
                [0.25, 0.75],
                [0.25, 0.75]
            ]
            self.prob_list_cues = np.repeat(np.expand_dims(self.prob_list_cues,0),batch_size, axis=0)
        elif self.mode == 'independent':
            self.prob_list_cues = np.random.uniform(size=(batch_size,2))
            self.prob_list_cues = np.expand_dims(self.prob_list_cues.round(2),1)

            self.number_cues = 1
            self.trials_per_cue = 96
        elif self.mode.startswith('exp'):
            if p_idx == -1: 
                raise ValueError('Please select a participant idx.')
            if self.mode.startswith('exp1'):
                df = pd.read_csv('dat/lefebvre_exp/exp1.csv')
            if self.mode == 'exp2':
                df = pd.read_csv('dat/lefebvre_exp/exp2.csv')
                self.success = 0.5
                self.fail = -0.5
            self.p_df = df[df.participant_idx == p_idx]

            self.prob_list_cues = [
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75]
            ]
            self.prob_list_cues = np.repeat(np.expand_dims(self.prob_list_cues,0),batch_size, axis=0)

        if self.mode.startswith('exp'):
            # create an array of of 24 times each cue number
            cues = np.expand_dims(self.p_df.cues.values, 0)
            # create multiple cue indicator lists of size batch_size 
            self.cue_order = np.repeat(cues, batch_size, axis=0)
            self.expected_rewards = self.prob_list_cues * self.success + (1 - self.prob_list_cues) * self.fail
        else:
            # create an array of of 24 times each cue number
            cues = np.expand_dims(np.repeat(np.arange(self.number_cues), self.trials_per_cue), 0)
            # create multiple cue indicator lists of size batch_size 
            self.cue_order = np.repeat(cues, batch_size, axis=0)
            # shuffle for each batch differently
            np.apply_along_axis(np.random.shuffle, 1, self.cue_order)
            self.expected_rewards = self.prob_list_cues * self.success + (1 - self.prob_list_cues) * self.fail

        self.max_steps = len(self.cue_order[0])
        self.step = 0
        self.finished = False
    
    def cue(self):
        return torch.from_numpy(self.cue_order[:, self.step])

    def sample(self, actions):
        cue = self.cue_order[:, self.step]

        if self.with_random:
            self.rand_actions = np.random.randint(0,2,self.batch_size)
            rewards, counter_rewards, rand_rewards = self.rewards(actions, cue)
            regrets, rand_regrets, optimal_actions, rand_optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = (torch.from_numpy(rand_rewards), torch.from_numpy(rand_regrets), torch.from_numpy(rand_optimal_actions), torch.from_numpy(self.rand_actions))

        else:
            rewards, counter_rewards = self.rewards(actions, cue)
            regrets, optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = 'none'

        if self.mode.startswith('exp'):
            trial_df = self.p_df[self.p_df.trials_idx==self.step] 
            """Set rewards to the one of Lefebrve experiment, if same actions was chosen, otherwise use the sampled ones."""
            #rewards_old = rewards.copy()
            #rewards[np.asarray(actions) == trial_df.actions.item()] = trial_df.rewards.item()
            #counter_rewards_old = counter_rewards.copy()
            #counter_rewards[np.asarray(self.counter_actions) == trial_df.actions.item()] = trial_df.rewards.item()

        self.step +=1
        if self.step == self.max_steps:
            self.finished = True

        return torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions), random_data, self.finished

    def rewards(self, actions, cue):
        cues = self.prob_list_cues[np.arange(self.batch_size), cue, :] # select the correct cue that is used for this experiment
        probs = cues[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        self.counter_actions = [int(not(a)) for a in actions]
        counter_probs = cues[np.arange(len(actions)), self.counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail

        if self.with_random:
            """Get some random rewards"""
            rand_probs = cues[np.arange(len(self.rand_actions)), self.rand_actions] # select the prob of the chosen action for a success
            rand_successes = np.random.binomial(1,rand_probs)
            rand_rewards = rand_successes * self.success + np.logical_not(rand_successes) * self.fail
            return rewards, counter_rewards, rand_rewards
        else:
            return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions, cue):
        expected_rewards = self.expected_rewards[np.arange(self.batch_size), cue] # select the correct cue that is used for this experiment
        regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(expected_rewards, axis=1)
        if self.mode == 'test' or self.mode.startswith('exp'):
            optimal_actions = np.where((cue == 0) | (cue == 3), actions, optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
        
        if self.with_random:
            rand_regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(self.rand_actions)), self.rand_actions]
            rand_optimal_actions = np.argmax(expected_rewards, axis=1)
            if self.mode == 'test' or self.mode.startswith('exp'):
                rand_optimal_actions = np.where((cue == 0) | (cue == 3), self.rand_actions, rand_optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
            return regrets, rand_regrets, optimal_actions, rand_optimal_actions
        else:
            return regrets, optimal_actions

o = OptimBiasTask(3, mode='test', with_random=True)
o.sample([1,1,0])

## REFRACTORED VERSION
## Not yet in use
class ContextualBanditTask():
    def __init__(self, batch_size, with_random, mode='', success=1., fail=-1.) -> None:
        self.number_cues = 4
        self.trials_per_cue = 24
        self.batch_size = batch_size
        self.mode = mode
        self.with_random = with_random
        self.success = success
        self.fail = fail
        self.mode_to_prob_cues = {
            'test': [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
            'high_prob': [[0.9, 0.8], [0.8, 0.9], [0.6, 0.9], [0.7, 0.8]],
            'low_prob': [[0.1, 0.2], [0.2, 0.1], [0.4, 0.1], [0.3, 0.2]],
            'test_2': [[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
        }
        self.setup_prob_cues()
        self.cue_order = self.generate_cue_order()
        # shuffle for each batch differently
        np.apply_along_axis(np.random.shuffle, 1, self.cue_order)
        self.expected_rewards = self.calculate_expected_rewards()

        self.max_steps = len(self.cue_order[0])
        self.step = 0
        self.finished = False

    def calculate_expected_rewards(self):
        return self.prob_list_cues * self.success + (1 - self.prob_list_cues) * self.fail

    def generate_cue_order(self):
        cue_order = np.tile(np.arange(self.number_cues), (self.batch_size, self.trials_per_cue))
        # TODO: remove below if the same
        # create an array of of 24 times each cue number
        cues = np.expand_dims(np.repeat(np.arange(self.number_cues), self.trials_per_cue), 0)
        # create multiple cue indicator lists of size batch_size 
        cue_order_test = np.repeat(cues, self.batch_size, axis=0)

        if np.array_equal(cue_order, cue_order_test):
            return cue_order
        else:
            assert('error')

    def setup_prob_cues(self):
        if self.mode == 'train':
            self.prob_list_cues = np.random.uniform(size=(self.batch_size,8))
            self.prob_list_cues = np.reshape(self.prob_list_cues, (self.batch_size,4,2))
            self.prob_list_cues = self.prob_list_cues.round(2)
        elif self.mode in self.mode_to_prob_cues:
            self.prob_list_cues = np.repeat(np.array([self.mode_to_prob_cues[self.mode]]), self.batch_size, axis=0) # maybe np.expand_dims(...,0)

    def cue(self):
        return torch.from_numpy(self.cue_order[:, self.step])

    def sample(self, actions):
        cue = self.cue_order[:, self.step]

        if self.with_random:
            self.rand_actions = np.random.randint(0,2,self.batch_size)
            rewards, counter_rewards, rand_rewards = self.rewards(actions, cue)
            regrets, rand_regrets, optimal_actions, rand_optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = (torch.from_numpy(rand_rewards), torch.from_numpy(rand_regrets), torch.from_numpy(rand_optimal_actions), torch.from_numpy(self.rand_actions))

        else:
            rewards, counter_rewards = self.rewards(actions, cue)
            regrets, optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = 'none'

        self.step +=1
        if self.step == self.max_steps:
            self.finished = True

        return torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions), random_data, self.finished

    def rewards(self, actions, cue):
        cues = self.prob_list_cues[np.arange(self.batch_size), cue, :] # select the correct cue that is used for this experiment
        probs = cues[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        self.counter_actions = [int(not(a)) for a in actions]
        counter_probs = cues[np.arange(len(actions)), self.counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail

        if self.with_random:
            """Get some random rewards"""
            rand_probs = cues[np.arange(len(self.rand_actions)), self.rand_actions] # select the prob of the chosen action for a success
            rand_successes = np.random.binomial(1,rand_probs)
            rand_rewards = rand_successes * self.success + np.logical_not(rand_successes) * self.fail
            return rewards, counter_rewards, rand_rewards
        else:
            return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions, cue):
        expected_rewards = self.expected_rewards[np.arange(self.batch_size), cue] # select the correct cue that is used for this experiment
        regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(expected_rewards, axis=1)
        if self.mode == 'test':
            optimal_actions = np.where((cue == 0) | (cue == 3), actions, optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
        
        if self.with_random:
            rand_regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(self.rand_actions)), self.rand_actions]
            rand_optimal_actions = np.argmax(expected_rewards, axis=1)
            if self.mode == 'test':
                rand_optimal_actions = np.where((cue == 0) | (cue == 3), self.rand_actions, rand_optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
            return regrets, rand_regrets, optimal_actions, rand_optimal_actions
        else:
            return regrets, optimal_actions


#o = ContextualBanditTask(3, mode='test_2', with_random=True)
#o.sample([1,1,0])



class LefebrveContextualBanditTask():

    def __init__(self, with_random, mode, participant_idx):
        self.with_random = with_random
        self.mode = mode
        self.participant_idx = participant_idx

        if participant_idx == -1: 
            raise ValueError('Please select a participant idx.')
        if self.mode.startswith('exp1'):
            df = pd.read_csv('dat/lefebvre_exp/exp1.csv')
            self.success = 0.5
            self.fail = 0.0
        if self.mode == 'exp2':
            df = pd.read_csv('dat/lefebvre_exp/exp2.csv')
            self.success = 0.5
            self.fail = -0.5
        self.p_df = df[df.participant_idx == participant_idx]

        self.prob_list_cues = [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75]
        ]
        self.prob_list_cues = np.repeat(np.expand_dims(self.prob_list_cues,0),batch_size, axis=0)

        # create an array of of 24 times each cue number
        cues = np.expand_dims(self.p_df.cues.values, 0)
        # create multiple cue indicator lists of size batch_size 
        self.cue_order = np.repeat(cues, batch_size, axis=0)
        self.expected_rewards = self.prob_list_cues * self.success + (1 - self.prob_list_cues) * self.fail
       
        self.max_steps = len(self.cue_order[0])
        self.step = 0
        self.finished = False
    
    def cue(self):
        return torch.from_numpy(self.cue_order[:, self.step])

    def sample(self, actions):
        cue = self.cue_order[:, self.step]

        if self.with_random:
            self.rand_actions = np.random.randint(0,2,self.batch_size)
            rewards, counter_rewards, rand_rewards = self.rewards(actions, cue)
            regrets, rand_regrets, optimal_actions, rand_optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = (torch.from_numpy(rand_rewards), torch.from_numpy(rand_regrets), torch.from_numpy(rand_optimal_actions), torch.from_numpy(self.rand_actions))

        else:
            rewards, counter_rewards = self.rewards(actions, cue)
            regrets, optimal_actions = self.regrets_and_optimal_actions(actions, cue)
            random_data = 'none'

            trial_df = self.p_df[self.p_df.trials_idx==self.step] 
            """Set rewards to the one of Lefebrve experiment, if same actions was chosen, otherwise use the sampled ones."""
            #rewards_old = rewards.copy()
            #rewards[np.asarray(actions) == trial_df.actions.item()] = trial_df.rewards.item()
            #counter_rewards_old = counter_rewards.copy()
            #counter_rewards[np.asarray(self.counter_actions) == trial_df.actions.item()] = trial_df.rewards.item()

        self.step +=1
        if self.step == self.max_steps:
            self.finished = True

        return torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions), random_data, self.finished

    def rewards(self, actions, cue):
        cues = self.prob_list_cues[np.arange(self.batch_size), cue, :] # select the correct cue that is used for this experiment
        probs = cues[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        self.counter_actions = [int(not(a)) for a in actions]
        counter_probs = cues[np.arange(len(actions)), self.counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail

        if self.with_random:
            """Get some random rewards"""
            rand_probs = cues[np.arange(len(self.rand_actions)), self.rand_actions] # select the prob of the chosen action for a success
            rand_successes = np.random.binomial(1,rand_probs)
            rand_rewards = rand_successes * self.success + np.logical_not(rand_successes) * self.fail
            return rewards, counter_rewards, rand_rewards
        else:
            return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions, cue):
        expected_rewards = self.expected_rewards[np.arange(self.batch_size), cue] # select the correct cue that is used for this experiment
        regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(expected_rewards, axis=1)
        optimal_actions = np.where((cue == 0) | (cue == 3), actions, optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
        
        if self.with_random:
            rand_regrets =  expected_rewards.max(axis=1) - expected_rewards[np.arange(len(self.rand_actions)), self.rand_actions]
            rand_optimal_actions = np.argmax(expected_rewards, axis=1)
            rand_optimal_actions = np.where((cue == 0) | (cue == 3), self.rand_actions, rand_optimal_actions) # for cue 0 and 3, both actions have the same probability, thus the selected actions is always the optimal action
            return regrets, rand_regrets, optimal_actions, rand_optimal_actions
        else:
            return regrets, optimal_actions

# %%
