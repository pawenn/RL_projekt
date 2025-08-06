import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


seeds = [7668, 6094, 6720, 4685, 5577, 1035, 5224, 6389, 9873, 1996]
algos = ['AEForward_2M_Detach', 'RI']
n_seeds = len(seeds)

# Read data from different runs
# This is the toy data, you can also build a proper loop over your own runs.
eval_dfs = []
for algo in algos:
    for seed in seeds:
        df = pd.read_csv(f'evaluation/results/{algo}/DQNAgent{"AEForward" if algo.startswith("AEForward") else algo}_training_log_seed{seed}.csv', sep=';')
        df['interval'] = df['Timestep'] // 1000
        aggregated = df.groupby('interval').agg({
            'TD_mean_abs': 'mean',  # or 'sum', 'max', etc.
            'TD_std': 'mean'  # or 'sum', 'max', etc.
        }).reset_index()
        aggregated["algo"] = algo
        aggregated["seed"] = seed
        aggregated = aggregated[aggregated['interval'] < aggregated['interval'].max()]
        eval_dfs.append(aggregated)
print("Read csv-files")
eval_df = pd.concat(eval_dfs, ignore_index=True)
print("Concatenated dataframes")

steps = np.unique(eval_df['interval'].values) * 1000

summary_df = eval_df.groupby(['interval', 'algo']).agg({
    'TD_mean_abs': 'mean',
    'TD_std': 'mean'
}).reset_index()

# Convert interval back to timesteps
summary_df['timestep'] = summary_df['interval'] * 1000

# Plot
plt.figure(figsize=(10, 6))
algos = summary_df['algo'].unique()

for algo in algos:
    algo_df = summary_df[summary_df['algo'] == algo]
    
    timesteps = algo_df['timestep']
    print(timesteps.to_list())
    mean_abs = algo_df['TD_mean_abs']
    std = algo_df['TD_std']
    
    # Plot mean
    plt.plot(timesteps, mean_abs, label=f"{algo} Mean TD_abs")
    
    # Plot ± std shaded area
    plt.fill_between(
        timesteps,
        mean_abs - std,
        mean_abs + std,
        alpha=0.3,
        label=f"{algo} ± TD_std"
    )

plt.xlabel("Time Steps")
plt.ylabel("TD Mean Abs ± TD Std (Mean Across Seeds)")
plt.title("Bias ± Variance per Interval Across Seeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/bias_variance_plot.png")
plt.show()