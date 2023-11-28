
# In-Context Learning Dynamics in LLMs Depend on the Properties of the Learning Task

In this project we compare the learning dynamics of LLMs (`llm`) with that of humans (`humans`) and Meta-RL agents (`meta-rl`) based on three bandit tasks (folders with respective code in brackets):
- Optimism Bias Task @[lefebvre2017]
- Confirmation Bias Task @[chambon2020]
- Agency Task @[chambon2020]

The Rescorla-Wagner models used for characterizing the learning dynamics are in `rw_model`.

## Abstract

We want to investigate the in-context learning dynamics of LLMs. For this purpose, we use three tasks that have been performed with humans. We find that LLMs show human-like behavior.

Why do they exhibit this behavior? We trained a Meta-RL agent on the same tasks. Meta-RL agents are known to produce the optimal solution. They are also less constrained than constrained learning models. Meta-RL is an idealized in-context system, and we use it with a similar architecture.

We find that Meta-RL exhibits similar learning dynamics. Thus, it seems that similar learning dynamics occur in LLMs and Meta-RL agents because it is the rational solution. 


## Code

For the LLM results, we used the public Claude-v1 API. The evaluations were conducted on a Slurm-based cluster with a multiple CPUs in Python using PyTorch. 
Additional analyses were carried out using NumPy, Pandas, and SciPy. Jupyter Notebooks, Matplotlib and Seaborn were used for plotting.


## References

Chambon, V., Théro, H., Vidal, M., Vandendriessche, H., Haggard, P., & Palminteri, S. (2020). Information about action outcomes differentially affects learning from self-determined versus imposed choices. Nature Human Behaviour, 4(10), Article 10. https://doi.org/10.1038/s41562-020-0919-5

Lefebvre, G., Lebreton, M., Meyniel, F., Bourgeois-Gironde, S., & Palminteri, S. (2017). Behavioural and neural characterization of optimistic reinforcement learning. Nature Human Behaviour, 1(4), 0067. https://doi.org/10.1038/s41562-017-0067
