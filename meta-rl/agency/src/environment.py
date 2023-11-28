"""
Contextual Bandit Class
"""

import numpy as np
import torch
import pandas as pd


class PartialTask():
    """ Env for a 2-armed bandit task with forced choice """

    def __init__(self, batch_size=int, success=1., fail=-1., test_reward_probs:str='original', testing=False) -> None:
        
        # Assign input parameters as properties
        self.batch_size = batch_size
        self.success = success
        self.fail = fail
        self.number_of_cues = 2
        self.test_reward_probs = test_reward_probs

        # Setup the correct acount of blocks/trials for the task
        if testing:
            self.number_free = 40 // 4
            self.number_forced = 80 // 4
        else:
            self.number_free = 40
            self.number_forced = 80
        self.number_of_blocks = 4
        self.total_trials = (2 * self.number_free) + (2 * self.number_forced)
        self.setup_blocks()
        self.start_block(0)

    def setup_blocks(self):
        """ Set up a dict containing positions of forced actions and trial order of forced actions in blocks. """
        reward_block_type = np.repeat([[0,0,1,1]], repeats=self.batch_size, axis=0) # high/low reward blocks 
        forced_type = np.array([0,1,0,1]) # free/forced choice blocks
        forced_actions, trial_order = self.setup_trial_order()
        self.blocks = {'reward_block_type': reward_block_type, # 0 high, 1 low
                       'forced_type':forced_type, # 0 free, 1 forced
                       'forced_actions': forced_actions, # 0: Symbol 1, 1: Symbol 2
                       'forced_trial_order': trial_order} # 0: free, 1: forced
        
        # Set up episode variables
        self.total_trials_idx = 0
        self.forced_block = 0

    def setup_trial_order(self):
        """ Determine the order of free vs forced choice tasks."""
        # actions that are taken in the forced-choice trials, 0: Symbol 1, 1: Symbol 2 (50/50)
        forced_actions = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks // 2, self.number_forced//4))
        np.apply_along_axis(np.random.shuffle, 2, forced_actions)
        # trial order of free-choice (0) and forced-choice trials (1) 
        trial_order = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks // 2, self.number_forced//2))
        np.apply_along_axis(np.random.shuffle, 2, trial_order)
        
        return forced_actions, trial_order

    def start_block(self, block, train=False):
        """ Set up the block for the next trial. """
        self.step_in_block = 0
        self.reward_block_type = self.blocks['reward_block_type'][:, block]
        self.forced_type = self.blocks['forced_type'][block]

        if self.forced_type == 1:
            self.forced_step_in_block = np.zeros(self.batch_size, dtype=int)
            self.forced_trial_order = self.blocks['forced_trial_order'][:, self.forced_block, :]
            self.forced_actions = self.blocks['forced_actions'][:, self.forced_block, :]
            self.forced_block += 1
            self.total_trials_block = self.number_forced
        else:
            self.total_trials_block = self.number_free

        # Set up reward probabilities
        if train: 
            self.rew_probs = np.random.uniform(size=(self.batch_size,2)).round(2)
        else:
            self.set_test_reward_probs()
        self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

    def set_test_reward_probs(self):

        if self.test_reward_probs == 'original':
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.6]),
                1: np.array([0.4, 0.1])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])
        elif self.test_reward_probs == 'high':
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.8]),
                1: np.array([0.8, 0.9])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])

    def cue(self, step):
        """ Returns batch wether trial is forced choice (1) or free choice (0) trial. """
        
        one_hot_cues = np.zeros((self.batch_size, 2))

        if self.forced_type == 1:
            cue = self.forced_trial_order[:, step]
            # Get the forced actions
            forced_trials_index = np.where(cue == 1)[0]

            if forced_trials_index.size > 0:
                forced_actions = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
                one_hot_forced_actions = np.zeros((len(forced_actions), 2))
                one_hot_forced_actions[np.arange(len(forced_actions)), forced_actions] = 1

                one_hot_cues[forced_trials_index, :] = one_hot_forced_actions
        
        return torch.from_numpy(one_hot_cues)

    def sample(self, actions, step):
        """ 
        Samples rewards and regrets. 
        
        Parameters:
            - actions: batch of actions in current bandit
            - step in current bandit

        """

        # Overwrite action in forced choice trials.
        actions = np.asarray(actions) if isinstance(actions, list) else actions

        if self.forced_type == 1:
            cue = self.forced_trial_order[:, step]
            forced_trials_index = np.where(cue == 1)[0]

            if forced_trials_index.size > 0:
                actions[forced_trials_index] = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
                self.forced_step_in_block[forced_trials_index] += 1

        rewards, counter_rewards = self.rewards(actions)
        regrets, optimal_actions = self.regrets_and_optimal_actions(actions)

        # set indices correctly
        self.total_trials_idx +=1
        self.step_in_block +=1
        return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)

    def rewards(self, actions):
        probs = self.rew_probs[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        counter_actions = [int(not(a)) for a in actions]
        counter_probs = self.rew_probs[np.arange(len(actions)), counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail
        return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions):
        regrets =  self.expected_rewards.max(axis=1) - self.expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(self.expected_rewards, axis=1)
        return regrets, optimal_actions


class ForcedChoiceTask():
    """ Env for a 2-armed bandit task with forced choice """

    def __init__(self, batch_size=int, agency=str, feedback=str, success=1., fail=-1., test=False, test_reward_probs:str='original', volatility=False, volatility_steps=20) -> None:
        
        # Assign input parameters as properties
        self.batch_size = batch_size
        self.success = success
        self.fail = fail
        self.number_of_cues = 2
        self.agency = agency
        self.feedback = feedback
        self.test = test
        self.test_reward_probs = test_reward_probs
        self.volatility = volatility
        self.volatility_steps = volatility_steps

        # Setup the correct acount of blocks/trials for the task        
        if self.feedback == 'partial':
            self.number_of_trials_free = 40
            self.number_of_trials_forced = 40 if self.agency == 'forced' else 0
            self.number_of_blocks = 12

        elif self.feedback == 'full':
            self.number_of_trials_free = 20 if self.agency == 'forced' else 40 # set to 40 in case of complete free-choice
            self.number_of_trials_forced = 20 if self.agency == 'forced' else 0
            self.number_of_blocks = 8

        if self.test:
            self.number_of_trials_free //= 2
            self.number_of_trials_forced //= 2
            self.number_of_blocks //= 3

        self.total_trials = self.number_of_trials_free + self.number_of_trials_forced

        self.setup_blocks()

    def set_test_reward_probs(self):

        if self.test_reward_probs == 'original':
            reward_block_type_dict = {0: 'high_reward', 1: 'low_reward'}
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.6]),
                1: np.array([0.4, 0.1])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])
        elif self.test_reward_probs == 'high':
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.8]),
                1: np.array([0.8, 0.9])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])

    def setup_blocks(self):
        """ Set up a dict containing positions of forced actions and trial order of forced actions in blocks. """
        reward_block_type = np.tile(np.arange(2), (self.batch_size, int(self.number_of_blocks/2))) # high / low reward blocks
        np.apply_along_axis(np.random.shuffle, 1, reward_block_type)
        forced_actions, trial_order = self.setup_trial_order()
        self.blocks = {'reward_block_type': reward_block_type, 'forced_actions': forced_actions, 'trial_order': trial_order}

    def setup_trial_order(self):
        """ Determine the order of free vs forced choice tasks."""
        if self.agency == 'forced':
            # actions that are taken in the forced-choice trials, 0: Symbol 1, 1: Symbol 2 (50/50)
            forced_actions = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks, int(self.number_of_trials_forced/2)))
            np.apply_along_axis(np.random.shuffle, 2, forced_actions)
            # trial order of free-choice (0) and forced-choice trials (1) 
            trial_order = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks, int(self.total_trials/2)))
            np.apply_along_axis(np.random.shuffle, 2, trial_order)
        else:
            forced_actions = 'none'
            trial_order = np.tile(0, (self.batch_size, self.number_of_blocks, self.total_trials))
            #np.apply_along_axis(np.random.shuffle, 2, trial_order)

        return forced_actions, trial_order

    def start_block(self, block, train=False):
        """ Set up the block for the next trial. """
        self.step_in_block = 0
        self.forced_step_in_block = np.zeros(self.batch_size, dtype=int)
        self.finished = False

        self.reward_block_type = self.blocks['reward_block_type'][:, block]
        self.trial_order = self.blocks['trial_order'][:, block, :]
        if self.agency == 'forced':
            self.forced_actions = self.blocks['forced_actions'][:, block, :]
        else:
            self.forced_actions = 'none'

        # Set up reward probabilities
        if train: 
            if self.test:
                first_prob = np.random.choice([0.1,0.9], size=(self.batch_size))
                second_prob = np.zeros_like(first_prob)
                second_prob = 1 - first_prob
                self.rew_probs = np.stack([first_prob, second_prob], axis=-1) 

            else:
                self.rew_probs = np.random.uniform(size=(self.batch_size,2)).round(2)

        else:
            self.set_test_reward_probs()
        
        self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

    def cue(self, step):
        """ Returns batch wether trial is forced choice (1) or free choice (0) trial. """
        
        cue = self.trial_order[:, step]
        # convert into encoding if cue is 0 then 00, else 10 if cue == 1 and action == 0, else 01 if cue == 1 and action == 1
        one_hot_cues = np.zeros((self.batch_size, 2))


        # Get the forced actions
        forced_trials_index = np.where(cue == 1)[0]
        if forced_trials_index.size > 0:
            forced_actions = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
            one_hot_forced_actions = np.zeros((len(forced_actions), 2))
            one_hot_forced_actions[np.arange(len(forced_actions)), forced_actions] = 1

            one_hot_cues[forced_trials_index, :] = one_hot_forced_actions
        
        return torch.from_numpy(one_hot_cues)

    def sample(self, actions, step):
        """ Samples rewards and regrets. """

        # Overwrite action in forced choice trials.
        actions = np.asarray(actions) if isinstance(actions, list) else actions
        cue = self.trial_order[:, step]
        forced_trials_index = np.where(cue == 1)[0]

        if forced_trials_index.size > 0:
            actions[forced_trials_index] = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
            self.forced_step_in_block[forced_trials_index] += 1


        if self.volatility and self.step_in_block % self.volatility_steps == 0:
            self.rew_probs = self.rew_probs[:, [1, 0]]
            self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

        rewards, counter_rewards = self.rewards(actions)
        regrets, optimal_actions = self.regrets_and_optimal_actions(actions)

        self.step_in_block +=1
        if self.step_in_block == self.total_trials:
            self.finished = True

        return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)

    def rewards(self, actions):
        probs = self.rew_probs[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        counter_actions = [int(not(a)) for a in actions]
        counter_probs = self.rew_probs[np.arange(len(actions)), counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail
        return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions):
        regrets =  self.expected_rewards.max(axis=1) - self.expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(self.expected_rewards, axis=1)
        return regrets, optimal_actions

class VolatilityTask():
    # TODO make it more specific to the volatility task
    """ Env for a 2-armed bandit task with forced choice """

    def __init__(self, batch_size=int, agency=str, feedback=str, success=1., fail=-1., test_reward_probs:str='original', volatility=False, volatility_steps=20) -> None:
        
        # Assign input parameters as properties
        self.batch_size = batch_size
        self.success = success
        self.fail = fail
        self.number_of_cues = 2
        self.agency = agency
        self.feedback = feedback
        self.test_reward_probs = test_reward_probs
        self.volatility = volatility
        self.volatility_steps = volatility_steps

        # Setup the correct acount of blocks/trials for the task        
        if self.feedback == 'partial':
            self.number_of_trials_free = 40
            self.number_of_trials_forced = 40 if self.agency == 'forced' else 0
            self.number_of_blocks = 12

        elif self.feedback == 'full':
            self.number_of_trials_free = 20 if self.agency == 'forced' else 40 # set to 40 in case of complete free-choice
            self.number_of_trials_forced = 20 if self.agency == 'forced' else 0
            self.number_of_blocks = 8

        if self.test:
            self.number_of_trials_free //= 2
            self.number_of_trials_forced //= 2
            self.number_of_blocks //= 3

        self.trials_per_block = self.number_of_trials_free + self.number_of_trials_forced

        self.setup_blocks()

    def set_test_reward_probs(self):

        if self.test_reward_probs == 'original':
            reward_block_type_dict = {0: 'high_reward', 1: 'low_reward'}
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.6]),
                1: np.array([0.4, 0.1])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])
        elif self.test_reward_probs == 'high':
            self.reward_block_prob_dict = {
                0: np.array([0.9, 0.8]),
                1: np.array([0.8, 0.9])
            }
            self.rew_probs = np.array([self.reward_block_prob_dict[batch_block] for batch_block in self.reward_block_type])

    def setup_blocks(self):
        """ Set up a dict containing positions of forced actions and trial order of forced actions in blocks. """
        reward_block_type = np.tile(np.arange(2), (self.batch_size, int(self.number_of_blocks/2))) # high / low reward blocks
        np.apply_along_axis(np.random.shuffle, 1, reward_block_type)
        forced_actions, trial_order = self.setup_trial_order()
        self.blocks = {'reward_block_type': reward_block_type, 'forced_actions': forced_actions, 'trial_order': trial_order}

    def setup_trial_order(self):
        """ Determine the order of free vs forced choice tasks."""
        if self.agency == 'forced':
            # actions that are taken in the forced-choice trials, 0: Symbol 1, 1: Symbol 2 (50/50)
            forced_actions = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks, int(self.number_of_trials_forced/2)))
            np.apply_along_axis(np.random.shuffle, 2, forced_actions)
            # trial order of free-choice (0) and forced-choice trials (1) 
            trial_order = np.tile(np.arange(2), (self.batch_size, self.number_of_blocks, int(self.trials_per_block/2)))
            np.apply_along_axis(np.random.shuffle, 2, trial_order)
        else:
            forced_actions = 'none'
            trial_order = np.tile(0, (self.batch_size, self.number_of_blocks, self.trials_per_block))
            #np.apply_along_axis(np.random.shuffle, 2, trial_order)

        return forced_actions, trial_order

    def start_block(self, block, train=False):
        """ Set up the block for the next trial. """
        self.step_in_block = 0
        self.forced_step_in_block = np.zeros(self.batch_size, dtype=int)
        self.finished = False

        self.reward_block_type = self.blocks['reward_block_type'][:, block]
        self.trial_order = self.blocks['trial_order'][:, block, :]
        if self.agency == 'forced':
            self.forced_actions = self.blocks['forced_actions'][:, block, :]
        else:
            self.forced_actions = 'none'

        # Set up reward probabilities
        if train: 
            if self.test:
                first_prob = np.random.choice([0.1,0.9], size=(self.batch_size))
                second_prob = np.zeros_like(first_prob)
                second_prob = 1 - first_prob
                self.rew_probs = np.stack([first_prob, second_prob], axis=-1) 

            else:
                self.rew_probs = np.random.uniform(size=(self.batch_size,2)).round(2)

        else:
            self.set_test_reward_probs()
        
        self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

    def cue(self, step):
        """ Returns batch wether trial is forced choice (1) or free choice (0) trial. """
        
        cue = self.trial_order[:, step]
        # convert into encoding if cue is 0 then 00, else 10 if cue == 1 and action == 0, else 01 if cue == 1 and action == 1
        one_hot_cues = np.zeros((self.batch_size, 2))


        # Get the forced actions
        forced_trials_index = np.where(cue == 1)[0]
        if forced_trials_index.size > 0:
            forced_actions = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
            one_hot_forced_actions = np.zeros((len(forced_actions), 2))
            one_hot_forced_actions[np.arange(len(forced_actions)), forced_actions] = 1

            one_hot_cues[forced_trials_index, :] = one_hot_forced_actions
        
        return torch.from_numpy(one_hot_cues)

    def sample(self, actions, step):
        """ Samples rewards and regrets. """

        # Overwrite action in forced choice trials.
        actions = np.asarray(actions) if isinstance(actions, list) else actions
        cue = self.trial_order[:, step]
        forced_trials_index = np.where(cue == 1)[0]

        if forced_trials_index.size > 0:
            actions[forced_trials_index] = self.forced_actions[forced_trials_index, self.forced_step_in_block[forced_trials_index]]
            self.forced_step_in_block[forced_trials_index] += 1


        if self.volatility and self.step_in_block % self.volatility_steps == 0:
            self.rew_probs = self.rew_probs[:, [1, 0]]
            self.expected_rewards = self.rew_probs * self.success + (1 - self.rew_probs) * self.fail

        rewards, counter_rewards = self.rewards(actions)
        regrets, optimal_actions = self.regrets_and_optimal_actions(actions)

        self.step_in_block +=1
        if self.step_in_block == self.trials_per_block:
            self.finished = True

        return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)

    def rewards(self, actions):
        probs = self.rew_probs[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self.success + np.logical_not(successes) * self.fail

        counter_actions = [int(not(a)) for a in actions]
        counter_probs = self.rew_probs[np.arange(len(actions)), counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self.success + np.logical_not(counter_successes) * self.fail
        return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions):
        regrets =  self.expected_rewards.max(axis=1) - self.expected_rewards[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(self.expected_rewards, axis=1)
        return regrets, optimal_actions


class TwoArmedBandit():
    def __init__(self, batch_size:int, success=1., fail=-1, dist='low', max_steps=100) -> None:
        """A stationary two-armed Bernoulli bandit"""

        self.max_steps = max_steps

        if dist == 'low':
            first_prob = np.random.choice([0.3,0.4], size=(batch_size, max_steps))
        elif dist == 'high':
            first_prob = np.random.choice([0.7,0.8], size=(batch_size, max_steps))
        else:
            raise ValueError('Distribution has to be either "high" or "low".')

        # Initialize an empty array with the same shape for the second values
        second_prob = np.zeros_like(first_prob)

        # Fill the second_values array according to your condition
        if dist == 'low':
            second_prob[first_prob == 0.3] = 0.4
            second_prob[first_prob == 0.4] = 0.3
        else:
            second_prob[first_prob == 0.8] = 0.7
            second_prob[first_prob == 0.7] = 0.8

        # Stack the two arrays along a new last dimension
        self.arms_p = np.stack([first_prob, second_prob], axis=-1)  # Shape (10, 100, 2)


        self._number_of_arms = len(self.arms_p[0]) # 2
        self._s = success
        self._f = fail
        self.batch_size = batch_size

        # calculates the expected reward per arm and saves it as an np.array
        self.expected_rewards = self.arms_p * self._s + (1 - self.arms_p) * self._f 

    def sample(self, actions, step):
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

        rewards, counter_rewards = self.rewards(actions, step)
        regrets, optimal_actions = self.regrets_and_optimal_actions(actions, step)

        return torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(counter_rewards), torch.from_numpy(regrets), torch.from_numpy(optimal_actions)

    def rewards(self, actions, step):
        prob = self.arms_p[:, step, :]
        probs = prob[np.arange(len(actions)), actions] # select the prob of the chosen action for a success
        successes = np.random.binomial(1,probs)
        rewards = successes * self._s + np.logical_not(successes) * self._f

        counter_actions = [int(not(a)) for a in actions]
        counter_probs = self.arms_p[np.arange(len(actions)), counter_actions] # select the prob of the chosen action for a success
        counter_successes = np.random.binomial(1,counter_probs)
        counter_rewards = counter_successes * self._s + np.logical_not(counter_successes) * self._f
        return rewards, counter_rewards

    def regrets_and_optimal_actions(self, actions, step):
        exp = self.expected_rewards[:, step, :]
        regrets =  exp.max(axis=1) - exp[np.arange(len(actions)), actions]
        optimal_actions = np.argmax(exp, axis=1)
        return regrets, optimal_actions

    def cue(self, step):    
        """ Retruns 0 only for test purposes."""    
        one_hot_cues = np.zeros((self.batch_size, 2))        
        return torch.from_numpy(one_hot_cues)

#bandit = TwoArmedBandit(batch_size=2, dist='low')
#bandit.sample(np.array([0,1]), 0)
#bandit.cue(0)