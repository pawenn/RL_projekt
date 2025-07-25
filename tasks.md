- per agent: 
    - init()
    - predict_agent()
    - update_agent()

- 3 subclasses of dqn_agent
    - raw-image agent (maybe?)
        - encoder as simple cnn
    - agent with AE and reconstruction
        - encoder + decoder + recon-loss
    - agent with AE and forward prediction
        - encoder + forward-model + pred-loss

- L2 regularization and z-weight decay (for both AEs?)

- (polyak-averaging?)
    - updating encoders and Q-heads at different rates



project structure:
    - networks
        - q_network.py +++
        - encoder.py +++
        - decoder.py +++
        - forward_model.py ---
    - buffer
        - abstract_buffer.py +++
        - buffers.py +++
    - dqn-agents
        - abstract_agent.py +++
        - dqn_agent.py (+++)
        - dqn_agent_RI.py (+++)
        - dqn_agent_AE_recon.py ---
        - dqn_agent_AE_forward.py ---
    - configs
        - dqn_agent_RI.yaml ---
        - dqn_agent_AE_recon.yaml ---
        - dqn_agent_AE_forward.yaml ---
    - results
        - dqn_agent_RI
        - dqn_agent_AE_recon
        - dqn_agent_AE_forward
    - util
        - FrameStackWrapper.py +++

logging:
    - every 1000 steps
        - 10 eval episodes (extra eval_env)
            - avg return
    - every step
        - measure TD error
    - one csv file for each metric (for each agent on a single seed)
