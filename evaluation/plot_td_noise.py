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
            'TD_std': 'mean'  # or 'sum', 'max', etc.
        }).reset_index()
        aggregated["algo"] = algo
        aggregated["seed"] = seed
        eval_dfs.append(aggregated)
print("Read csv-files")
eval_df = pd.concat(eval_dfs, ignore_index=True)
print("Concatenated dataframes")

steps = np.unique(eval_df['interval'].values) * 1000

# You can add other algorithms here
train_scores = {
    "Raw-Image": eval_df.loc[eval_df['algo'] == 'RI']["TD_std"].to_numpy().reshape((n_seeds, -1)),
    "AE-Forward": eval_df.loc[eval_df['algo'] == 'AEForward_2M_Detach']["TD_std"].to_numpy().reshape((n_seeds, -1)),
}
print("Reshaped dataframes")

# This aggregates only IQM, but other options include mean and median
# Optimality gap exists, but you obviously need optimal scores for that
# If you want to use it, check their code
iqm = lambda scores: np.array(  # noqa: E731
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
median = lambda scores: np.median(scores, axis=0)
mean = lambda scores: np.mean(scores, axis=0)

iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)
print("Computed IQM scores")


# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps,
    iqm_scores,
    iqm_cis,
    algorithms=["Raw-Image", "AE-Forward"],
    xlabel="Time Steps",
    ylabel="IQM Standard Deviation of TD-Error per Update-Batch",
)
plt.gcf().canvas.manager.set_window_title("IQM TD-noise")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/td_noise_IQM.png")
plt.savefig("evaluation/plots/td_noise/td_noise_IQM.pdf")
plt.savefig("evaluation/plots/td_noise/td_noise_IQM.svg")
plt.show()



median_scores, median_cis = get_interval_estimates(
    train_scores,
    median,
    reps=2000,
)
print("Computed median scores")

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps,
    median_scores,
    median_cis,
    algorithms=["Raw-Image", "AE-Forward"],
    xlabel="Time Steps",
    ylabel="Median Standard Deviation of TD-Error per Update-Batch",
)
plt.gcf().canvas.manager.set_window_title("Median TD-noise")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/td_noise_median.png")
plt.savefig("evaluation/plots/td_noise/td_noise_median.pdf")
plt.savefig("evaluation/plots/td_noise/td_noise_median.svg")
plt.show()



mean_scores, mean_cis = get_interval_estimates(
    train_scores,
    mean,
    reps=2000,
)
print("Computed mean scores")

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps,
    mean_scores,
    mean_cis,
    algorithms=["Raw-Image", "AE-Forward"],
    xlabel="Time Steps",
    ylabel="Mean Standard Deviation of TD-Error per Update-Batch",
)
plt.gcf().canvas.manager.set_window_title("Mean TD-noise")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/td_noise_mean.png")
plt.savefig("evaluation/plots/td_noise/td_noise_mean.pdf")
plt.savefig("evaluation/plots/td_noise/td_noise_mean.svg")
plt.show()
