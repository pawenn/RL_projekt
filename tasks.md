- TODO:

    - save agent after training                     Jakob
        - override save/load methods in agents
    - two encoder-optimizers in recon-agent?                Paul
    - FrameSkip only for terminated, but not truncated?     Paul
        - also 1 time-step can be less than 4 env-steps
    - logging per episode or step?                          Paul
        - currently: TD-noise = avg of all td_noise_episode (including prior episodes)
        - maybe:
            - csv-train per train-step (~ 4 * env-frame):
                - Time-step; Frame; Episode; Reward; Loss; Aux-Loss; TD_mean; TD_std; TD_max; TD_min; Time
            - csv-train per train-episode:
                - Time-step; Frame; Episode; Ep-Reward; Time
            - csv-eval per eval-interval:
                - Time-step; Frame; Eval-Episode; Eval-Reward (per current eval-epsiode); Avg-Episode-Length
    - requirements.txt          Jakob
    - seeds for experiments     Jakob

    - script for visualization



- (polyak-averaging?)
    - updating encoders and Q-heads at different rates  later


