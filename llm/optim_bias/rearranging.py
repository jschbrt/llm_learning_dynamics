import numpy as np
import pandas as pd
import glob


#! Rearranged human data to be in similar format as the other engines for plotting purposes
# path = 'data/humans/exp1.csv'
# df = pd.read_csv(path)
# # Make rewards that are equal to 0.5 into 1
# df['rewards'] = df['rewards'].replace(0.5, 1)

# # Rename actions column to choices and add 1
# df = df.rename(columns={'actions': 'choice'})

# # Rename cues to casinos and add 1
# df = df.rename(columns={'cues': 'casino'})
# df['casino'] += 1

# # Rename trials_idx to trial
# df = df.rename(columns={'trials_idx': 'trial'})

# # Restore in 'data/humans' as f'run_{participant_idx} for participant_idx in range(0, 50)
# for participant_idx in range(0, 50):
#     df_participant = df[df['participant_idx'] == participant_idx]
#     # Reset index and get rid of unnamed column
#     df_participant = df_participant.reset_index()
#     df_participant = df_participant.drop(columns=['index', 'Unnamed: 0'])
#     df_participant.to_csv(f'data/humans/run_{participant_idx}.csv', index=False)


#! Add rewards to the data
# path = 'data/llama_65'
# path = 'data/claude-v1'
# for engine in [ "llama_30", "gpt-4", "vicuna_13", "vicuna_7","vicuna_7", "text-davinci-003", "debugging"]:
#     path = f'data/{engine}'
#     runs = len(glob.glob(f'{path}/run_*.csv'))
#     for run in range(runs):
#         df = pd.read_csv(f'{path}/run_' + str(run) + '.csv')
#         rewards = []
#         for idx, row in df.iterrows():
#             if row['choice'] == 0:
#                 rewards.append(row['reward0'])
#             else:
#                 rewards.append(row['reward1'])

#         df['rewards'] = rewards
#         df.to_csv(f'{path}/run_' + str(run) + '.csv', index=False)

#! change rewards to scalar from list 
# for engine in [ "text-bison", 'claude-2', 'hf_falcon-40b', 'hf_falcon-40b-instruct', 'hf_mpt-30b', 'hf_mpt-30b-instruct', 'hf_mpt-30b-chat']:
for engine in [ "llama_13", "llama_7"]:
    path = f'data/{engine}'
    runs = len(glob.glob(f'{path}/run_*.csv'))
    for run in range(runs): 
        # Change the rewards which right now is in the form "[rewards_trial0, rewards_trial1, ... ,rewards_trial95]" to "rewards_trial<the trial the row is in>"
        df = pd.read_csv(f'{path}/run_' + str(run) + '.csv')
        rewards = eval(df['rewards'][0]) #All the rewards are the same so just take the first one
        df['rewards'] = rewards
        df.to_csv(f'{path}/run_' + str(run) + '.csv', index=False)

