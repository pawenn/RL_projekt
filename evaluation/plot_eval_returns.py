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


# Read data; Group by evaluation-interval and compute avg return
eval_dfs = []
for algo in algos:
    for seed in seeds:
        df = pd.read_csv(f'evaluation/results/{algo}/DQNAgent{algo}_eval_log_seed{seed}.csv', sep=';')
        df = df[['Eval-Interval', 'Episode-Reward']].groupby('Eval-Interval').mean()
        df["algo"] = algo
        df["seed"] = seed
        eval_dfs.append(df)
eval_df = pd.concat(eval_dfs, ignore_index=True)

steps = (np.arange(24) + 1) * 1250


# Get scores for each algorithm
train_scores = {
    "Raw-Image": eval_df.loc[eval_df['algo'] == 'RI']["Episode-Reward"].to_numpy().reshape((n_seeds, -1)),
    "AE-Reconstruction": eval_df.loc[eval_df['algo'] == 'AERecon']["Episode-Reward"].to_numpy().reshape((n_seeds, -1)),
    "AE-Forward-Prediction": eval_df.loc[eval_df['algo'] == 'AEForward']["Episode-Reward"].to_numpy().reshape((n_seeds, -1)),
}


# Metrics
iqm = lambda scores: np.array(
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
median = lambda scores: np.median(scores, axis=0)
mean = lambda scores: np.mean(scores, axis=0)



# IQM
iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

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
    ylabel="IQM of Avg Evaluation Return",
)
plt.gcf().canvas.manager.set_window_title("IQM of Average Return for5 Evaluation Episodes (across 10 Seeds)")
plt.title("IQM of Average Return for\n5 Evaluation Episodes (across 10 Seeds)", fontsize='xx-large')
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/eval_returns/eval_return_IQM.png")
plt.savefig("evaluation/plots/eval_returns/eval_return_IQM.pdf")
plt.savefig("evaluation/plots/eval_returns/eval_return_IQM.svg")
plt.show()



# Median
median_scores, median_cis = get_interval_estimates(
    train_scores,
    median,
    reps=2000,
)

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
    ylabel="Median of Avg Evaluation Return",
)
plt.gcf().canvas.manager.set_window_title("Median of Average Return for 5 Evaluation Episodes (across 10 Seeds)")
plt.title("Median of Average Return for\n5 Evaluation Episodes (across 10 Seeds)", fontsize='xx-large')
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/eval_returns/eval_return_median.png")
plt.savefig("evaluation/plots/eval_returns/eval_return_median.pdf")
plt.savefig("evaluation/plots/eval_returns/eval_return_median.svg")
plt.show()



# Mean
mean_scores, mean_cis = get_interval_estimates(
    train_scores,
    mean,
    reps=2000,
)

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
    ylabel="Mean of Avg Evaluation Return",
)
plt.gcf().canvas.manager.set_window_title("Mean of Average Return for 5 Evaluation Episodes (across 10 Seeds)")
plt.title("Mean of Average Return for\n5 Evaluation Episodes (across 10 Seeds)", fontsize='xx-large')
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/plots/eval_returns/eval_return_mean.png")
plt.savefig("evaluation/plots/eval_returns/eval_return_mean.pdf")
plt.savefig("evaluation/plots/eval_returns/eval_return_mean.svg")
plt.show()
