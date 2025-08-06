import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


seeds = [7668, 6094, 6720, 4685, 5577, 1035, 5224, 6389, 9873, 1996]
algos = ['AERecon', 'AEForward',]
algo_names = ["AE-Reconstruction", "AE-Forward-Prediction"]
n_seeds = len(seeds)
eval_dfs = []


# Read data; Compute aggregation-intervals (1000 steps to remove noise from update of target-network)
for algo in algos:
    for seed in seeds:
        df = pd.read_csv(f'evaluation/results/{algo}/DQNAgent{algo}_training_log_seed{seed}.csv', sep=';')
        # Aggregate for intervals of 1000 steps (same as target network update frequency)
        df['interval'] = (df['Timestep'] - 1) // 1000  # -1, so that the last step is not its own interval
        aggregated = df.groupby('interval').agg({
            'Aux-Loss': 'mean',
        }).reset_index()
        aggregated["algo"] = algo
        aggregated["seed"] = seed
        eval_dfs.append(aggregated)

print("Read csv-files")
eval_df = pd.concat(eval_dfs, ignore_index=True)
print("Concatenated dataframes")

steps = (np.arange(30) + 1) * 1000


# Get aux-loss-values for each algorithm
aux_loss_values = {
    "AE-Reconstruction": eval_df.loc[eval_df['algo'] == 'AERecon']["Aux-Loss"].to_numpy().reshape((n_seeds, -1)),
    "AE-Forward-Prediction": eval_df.loc[eval_df['algo'] == 'AEForward']["Aux-Loss"].to_numpy().reshape((n_seeds, -1)),
}


print("Reshaped dataframes")


# Metrics
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
median = lambda scores: np.median(scores, axis=0)
mean = lambda scores: np.mean(scores, axis=0)



# IQM - Aux-Loss
iqm_scores, iqm_cis = get_interval_estimates(
    aux_loss_values,
    iqm,
    reps=2000,
)
print("Computed IQM scores")

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for i, algo in enumerate(algo_names):
    color = sns.color_palette("colorblind")[i+1]
    axs[i].plot(steps, iqm_scores[algo], label=algo, color=color, marker='o')
    lower, upper = iqm_cis[algo]
    axs[i].fill_between(steps, lower, upper, color=color, alpha=0.2)
    axs[i].set_ylabel("Aux-Loss", fontsize='xx-large')
    axs[i].legend()
    axs[i].grid(True)

axs[1].set_xlabel("Time Steps", fontsize='xx-large')
fig.suptitle("IQM of Average Auxiliary Loss\nper 1000 Steps (across 10 Seeds)", fontsize='xx-large')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("evaluation/plots/aux_loss/aux_loss_IQM_split.pdf")
plt.savefig("evaluation/plots/aux_loss/aux_loss_IQM_split.png")
plt.savefig("evaluation/plots/aux_loss/aux_loss_IQM_split.svg")
plt.show()



# Median - Aux-Loss
median_scores, median_cis = get_interval_estimates(
    aux_loss_values,
    median,
    reps=2000,
)
print("Computed median scores")

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for i, algo in enumerate(algo_names):
    color = sns.color_palette("colorblind")[i+1]
    axs[i].plot(steps, median_scores[algo], label=algo, color=color, marker='o')
    lower, upper = median_cis[algo]
    axs[i].fill_between(steps, lower, upper, color=color, alpha=0.2)
    axs[i].set_ylabel("Aux-Loss", fontsize='xx-large')
    axs[i].legend()
    axs[i].grid(True)

axs[1].set_xlabel("Time Steps", fontsize='xx-large')
fig.suptitle("Median of Average Auxiliary Loss\nper 1000 Steps (across 10 Seeds)", fontsize='xx-large')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("evaluation/plots/aux_loss/aux_loss_median_split.pdf")
plt.savefig("evaluation/plots/aux_loss/aux_loss_median_split.png")
plt.savefig("evaluation/plots/aux_loss/aux_loss_median_split.svg")
plt.show()



# Mean - Aux-Loss
mean_scores, mean_cis = get_interval_estimates(
    aux_loss_values,
    mean,
    reps=2000,
)
print("Computed mean scores")

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for i, algo in enumerate(algo_names):
    color = sns.color_palette("colorblind")[i+1]
    axs[i].plot(steps, mean_scores[algo], label=algo, color=color, marker='o')
    lower, upper = mean_cis[algo]
    axs[i].fill_between(steps, lower, upper, color=color, alpha=0.2)
    axs[i].set_ylabel("Aux-Loss", fontsize='xx-large')
    axs[i].legend()
    axs[i].grid(True)

axs[1].set_xlabel("Time Steps", fontsize='xx-large')
fig.suptitle("Mean of Average Auxiliary Loss\nper 1000 Steps (across 10 Seeds)", fontsize='xx-large')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("evaluation/plots/aux_loss/aux_loss_mean_split.pdf")
plt.savefig("evaluation/plots/aux_loss/aux_loss_mean_split.png")
plt.savefig("evaluation/plots/aux_loss/aux_loss_mean_split.svg")
plt.show()
