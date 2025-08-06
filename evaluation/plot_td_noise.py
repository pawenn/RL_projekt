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


# Read data; Compute aggregation-intervals (1000 steps to remove noise from update of target-network)
for algo in algos:
    for seed in seeds:
        df = pd.read_csv(f'evaluation/results/{algo}/DQNAgent{algo}_training_log_seed{seed}.csv', sep=';')
        # Aggregate for intervals of 1000 steps (same as target network update frequency)
        df['interval'] = (df['Timestep'] - 1) // 1000  # -1, so that the last step is not its own interval
        aggregated = df.groupby('interval').agg({
            'TD_std': 'mean'
        }).reset_index()

        aggregated["algo"] = algo
        aggregated["seed"] = seed
        eval_dfs.append(aggregated)

print("Read csv-files")
eval_df = pd.concat(eval_dfs, ignore_index=True)
print("Concatenated dataframes")

steps = (np.arange(30) + 1) * 1000


# Get std-values of td-error for each algorithm
td_error_std_values = {
    "Raw-Image": eval_df.loc[eval_df['algo'] == 'RI']["TD_std"].to_numpy().reshape((n_seeds, -1)),
    "AE-Reconstruction": eval_df.loc[eval_df['algo'] == 'AERecon']["TD_std"].to_numpy().reshape((n_seeds, -1)),
    "AE-Forward-Prediction": eval_df.loc[eval_df['algo'] == 'AEForward']["TD_std"].to_numpy().reshape((n_seeds, -1)),
}
print("Reshaped dataframes")


# Metrics
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
median = lambda scores: np.median(scores, axis=0)
mean = lambda scores: np.mean(scores, axis=0)



# IQM - STD
iqm_scores, iqm_cis = get_interval_estimates(
    td_error_std_values,
    iqm,
    reps=2000,
)
print("Computed IQM scores")

plot_sample_efficiency_curve(
    steps,
    iqm_scores,
    iqm_cis,
    algorithms=[
        "Raw-Image",
        "AE-Reconstruction",
        "AE-Forward-Prediction",
    ],
    xlabel="Time Steps",
    ylabel="IQM of Avg TD-Error-STD",
)
plt.gcf().canvas.manager.set_window_title("IQM TD-noise")
plt.title("IQM of Average TD-Error Standard Deviation\nfor 1000 Batch-Updates (across 10 Seeds)", fontsize='xx-large')
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/td_noise_IQM.png")
plt.savefig("evaluation/plots/td_noise/td_noise_IQM.pdf")
plt.savefig("evaluation/plots/td_noise/td_noise_IQM.svg")
plt.show()



# Median - STD
median_scores, median_cis = get_interval_estimates(
    td_error_std_values,
    median,
    reps=2000,
)
print("Computed median scores")

plot_sample_efficiency_curve(
    steps,
    median_scores,
    median_cis,
    algorithms=[
        "Raw-Image",
        "AE-Reconstruction",
        "AE-Forward-Prediction",
    ],
    xlabel="Time Steps",
    ylabel="Median of Avg TD-Error-STD",
)
plt.gcf().canvas.manager.set_window_title("Median TD-noise")
plt.title("Median of Average TD-Error Standard Deviation\nfor 1000 Batch-Updates (across 10 Seeds)", fontsize='xx-large')
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/td_noise_median.png")
plt.savefig("evaluation/plots/td_noise/td_noise_median.pdf")
plt.savefig("evaluation/plots/td_noise/td_noise_median.svg")
plt.show()



# Mean - STD
mean_scores, mean_cis = get_interval_estimates(
    td_error_std_values,
    mean,
    reps=2000,
)
print("Computed mean scores")

plot_sample_efficiency_curve(
    steps,
    mean_scores,
    mean_cis,
    algorithms=[
        "Raw-Image",
        "AE-Reconstruction",
        "AE-Forward-Prediction",
    ],
    xlabel="Time Steps",
    ylabel="Mean of Avg TD-Error-STD",
)
plt.gcf().canvas.manager.set_window_title("Mean TD-noise")
plt.title("Mean of Average TD-Error Standard Deviation\nfor 1000 Batch-Updates (across 10 Seeds)", fontsize='xx-large')
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/td_noise/td_noise_mean.png")
plt.savefig("evaluation/plots/td_noise/td_noise_mean.pdf")
plt.savefig("evaluation/plots/td_noise/td_noise_mean.svg")
plt.show()
