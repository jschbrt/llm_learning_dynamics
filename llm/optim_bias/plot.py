import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def get_rewards(engine):
    runs = len(glob.glob(f'data/{engine}/run_*.csv')) 
    no_trials = 96
    rewards = np.zeros((no_trials, runs))
    for run in range(runs):
        df = pd.read_csv(f'data/{engine}/run_' + str(run) + '.csv')
        for trial in range(no_trials):
            if engine == 'humans':
                rewards[trial, run] = df['rewards'][trial]
            else:
                rewards[trial, run] = df['reward0'][trial] if df['choice'][trial] == 0 else df['reward1'][trial]
    return rewards

def plot_reward_across_trials(engine_names, rewards):
    for count, engine in enumerate(engine_names):
        rewards_mean = np.mean(rewards[count], axis=1)
        rewards_ci =1.96* (np.std(rewards[count], axis=1)/ np.sqrt(np.shape(rewards[count][1]))) 
        plt.plot(rewards_mean, label=engine)
        plt.fill_between(range(len(rewards_mean)), rewards_mean - rewards_ci, rewards_mean + rewards_ci, alpha=0.2)
    plt.legend()
    plt.savefig('./plots/reward_across_trials.png')

def plot_avg_rewards(engine_names, rewards):
    for count, engine in enumerate(engine_names):
        rewards_mean = np.mean(rewards[count])
        rewards_ci = 1.96* np.std(rewards[count])/np.sqrt(np.shape(rewards[count][1])[0]*np.shape(rewards[count][0])[0])
        #barplot
        plt.bar(engine, rewards_mean, yerr=rewards_ci, capsize=4)
    # Plot horizontal line at 0.5
    plt.axhline(y=0.5, color='grey', linestyle='--', label= 'chance')
    plt.axhline(y=0.625, color='red', linestyle='--', label= 'ground truth')
    #Rotate xticks
    plt.xticks(rotation=45)
    #Extend frame to include xticks
    plt.subplots_adjust(bottom=0.2)
    plt.legend()
    plt.ylim(0.495, 1)
    plt.savefig('./plots/reward_avg.png')

if __name__ == "__main__":
    # engine_names = ["llama_65", "llama_30", "llama_13", "llama_7", "vicuna_7", "vicuna_13",  "text-davinci-002", "claude-v1"]
    rewards = []
    engine_names = [   "text-davinci-003", "claude-v1", 'humans', 'llama_65', 'debugging', 'vicuna_7', 'gpt-4', 'llama_30', 'text-davinci-002', 'vicuna_13', 'llama_7']
    # engine_names = ["claude-v1", 'humans', 'gpt-4', 'debugging']
    for engine in engine_names:
        print(engine)
        rewards.append(get_rewards(engine))
        if engine == 'llama_65':
            print(np.mean(rewards[-1], axis=0))
    plot_reward_across_trials(engine_names, rewards)
    plt.clf()
    plot_avg_rewards(engine_names, rewards)

