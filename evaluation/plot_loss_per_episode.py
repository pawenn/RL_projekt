import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


seeds = [7668, 6094, 6720, 4685, 5577, 1035, 5224, 6389, 9873, 1996]
algos = [
    'RI',
    'AERecon',
    'AEForward',
]
n_seeds = len(seeds)
eval_dfs = []
min_num_episodes = np.inf

# Read data;  Group by episode and compute avg loss
for algo in algos:
    for seed in seeds:
        df = pd.read_csv(f'evaluation/results/{algo}/DQNAgent{algo}_training_log_seed{seed}.csv', sep=';')
        aggregated = df.groupby('Episode').agg({
            'Loss': 'mean',
        }).reset_index()
        aggregated["algo"] = algo
        aggregated["seed"] = seed
        min_num_episodes = min(min_num_episodes, aggregated['Episode'].max())
        eval_dfs.append(aggregated)
print("Read csv-files")

eval_dfs = [df[df['Episode'] < min_num_episodes] for df in eval_dfs]
eval_df = pd.concat(eval_dfs, ignore_index=True)
print("Concatenated dataframes")

episodes = (np.arange(min_num_episodes) + 1)


# Get loss-values for each algorithm
loss_values = {
    "Raw-Image": eval_df.loc[eval_df['algo'] == 'RI']["Loss"].to_numpy().reshape((n_seeds, -1)),
    "AE-Reconstruction": eval_df.loc[eval_df['algo'] == 'AERecon']["Loss"].to_numpy().reshape((n_seeds, -1)),
    "AE-Forward-Prediction": eval_df.loc[eval_df['algo'] == 'AEForward']["Loss"].to_numpy().reshape((n_seeds, -1)),
}


print("Reshaped dataframes")


# Metrics
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
median = lambda scores: np.median(scores, axis=0)
mean = lambda scores: np.mean(scores, axis=0)



# IQM - Loss
iqm_scores, iqm_cis = get_interval_estimates(
    loss_values,
    iqm,
    reps=2000,
)
print("Computed IQM scores")

plot_sample_efficiency_curve(
    episodes,
    iqm_scores,
    iqm_cis,
    algorithms=[
        "Raw-Image",
        "AE-Reconstruction",
        "AE-Forward-Prediction",
    ],
    xlabel="Time Steps",
    ylabel="IQM of Avg Loss per Episode",
)
plt.gcf().canvas.manager.set_window_title("IQM of Average Loss per Episode (across 10 Seeds)")
plt.title("IQM of Average Loss per Episode (across 10 Seeds)")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/loss_per_episode/loss_IQM.png")
plt.savefig("evaluation/plots/loss_per_episode/loss_IQM.pdf")
plt.savefig("evaluation/plots/loss_per_episode/loss_IQM.svg")
plt.show()



# Median - Loss
median_scores, median_cis = get_interval_estimates(
    loss_values,
    median,
    reps=2000,
)
print("Computed median scores")

plot_sample_efficiency_curve(
    episodes,
    median_scores,
    median_cis,
    algorithms=[
        "Raw-Image",
        "AE-Reconstruction",
        "AE-Forward-Prediction",
    ],
    xlabel="Time Steps",
    ylabel="Median of Avg Loss per Episode",
)
plt.gcf().canvas.manager.set_window_title("Median of Average Loss per Episode (across 10 Seeds)")
plt.title("Median of Average Loss per Episode (across 10 Seeds)")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/loss_per_episode/loss_median.png")
plt.savefig("evaluation/plots/loss_per_episode/loss_median.pdf")
plt.savefig("evaluation/plots/loss_per_episode/loss_median.svg")
plt.show()



# Mean - Loss
mean_scores, mean_cis = get_interval_estimates(
    loss_values,
    mean,
    reps=2000,
)
print("Computed mean scores")

plot_sample_efficiency_curve(
    episodes,
    mean_scores,
    mean_cis,
    algorithms=[
        "Raw-Image",
        "AE-Reconstruction",
        "AE-Forward-Prediction",
    ],
    xlabel="Time Steps",
    ylabel="Mean of Avg Loss per Episode",
)
plt.gcf().canvas.manager.set_window_title("Mean of Average Loss per Episode (across 10 Seeds)")
plt.title("Mean of Average Loss per Episode (across 10 Seeds)")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/loss_per_episode/loss_mean.png")
plt.savefig("evaluation/plots/loss_per_episode/loss_mean.pdf")
plt.savefig("evaluation/plots/loss_per_episode/loss_mean.svg")
plt.show()
