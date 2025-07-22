- TODO:
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


- (polyak-averaging?)
    - updating encoders and Q-heads at different rates  later


