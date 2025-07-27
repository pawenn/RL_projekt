- TODO:

    - save agent after training
        - override save/load methods in agents
    - two encoder-optimizers in recon-agent?
    - FrameSkip only for terminated, but not truncated?
        - also 1 time-step can be less than 4 env-steps
    - logging per episode or step?
        - currently: TD-noise = avg of all td_noise_episode (including prior episodes)
        - maybe:
            - csv-train per time-step:
                - Time-step; Frame; Episode; Reward; Loss; Aux-Loss; TD_mean; TD_std; TD_max; TD_min; Time
            - csv-eval per eval-interval:
                - Time-step; Frame; Eval-Episode; Avg-Reward; Avg-Episode-Length
    - requirements.txt
    - seeds for experiments



    - seed !!!              Jakob

    - logging:              Paul
        - add these to second csv-file
            - TD-error:
                - in update-agent()
                    - compute TD-error -> log mean, std, max, min (maybe in dict?)
                        - how to other people to this / how do they measure TD-noise (loss)?
            - add loss
                add aux-loss
        - to first csv-file add:
            - avg reward of last 10 episodes

    - frame-skip:               Paul
        - Wrapper before FrameStack-Wrapper?

    - CNN-layers:           Paul
        - reduce num-layers? 

    - forward prediction:       Jakob
        - keep L2 regularization?
        - detach z_next?
        - add norm + tanh to forward-model?

    - update requirements.txt


- (polyak-averaging?)
    - updating encoders and Q-heads at different rates  later


