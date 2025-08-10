# Model-Free TD Learning in DQN:<br>A Comparison of Reconstruction and Forward Prediction

This repository contains the code used for training and evaluating the three DQN-agents on the CarRacing-v3 environment.
These work either only with the raw-images through a CNN, or with an auxiliary loss from an auto-encoder, namely through reconstruction or forward-prediction.


## Setup

- Python: 3.11.13
- Create a virtual environment, for example with conda (recommended):
    - `conda create -n <venv_name> python=3.11.13`
    - `conda activate <venv_name>`
    - `pip install -r requirements.txt`


## Project-Structure

Large parts of the project code are reused from the lecture. Though some parts were taken from other repositories, which are marked below.

### DQN-Agents

This directory holds the three main DQN-agent-classes:
- `DQNAgentRI`: The agent working only with the raw-image.
- `DQNAgentAErecon`: The agent using reconstruction as an auxiliary-loss.
- `DQNAgentAEForward`: The agent using forward-prediction as an auxiliary-loss.

To train one of the agents, simply run the corresponding `dqn_agent_*.py` file.

Also included are the base classes `AbstractAgent` and `DQNAgent`, from which the three other agents inherit.

### Configs

These .yaml-files contain the hyperparameter-configurations for training the agents.

### Networks

This directory includes the neural networks used by the agent:
- `QNetwork`: The network-class used to calculate the Q-values.
- `PixelEncoder`: The encoder-class used by all three agents, be it as a CNN for only working with the frames, or as an encoder for reconstruction or forward-prediction (taken from [SAC-AE](https://github.com/denisyarats/pytorch_sac_ae)).
- `PixelDecoder`: The decoder-class used by the reconstruction-agent (taken from [SAC-AE](https://github.com/denisyarats/pytorch_sac_ae)).
- `ForwardModel`: The network used by the agent performing forward-prediction.

### Buffer

Includes the `AbstractBuffer` base-class and the `ReplayBuffer` class, that is used by the DQN-agents.

### Utils

This directory contains environment-wrappers used for training all three agents:
- `SkipFrame`: Used for repeating one action for several steps to speed up and stabilize training (taken from [DQN-Car-Racing](https://github.com/wiitt/DQN-Car-Racing))
- `FrameStack`: Used for stacking several consecutive frames, so that the agents can better estimate the environment dynamics (taken from [SAC-AE](https://github.com/denisyarats/pytorch_sac_ae)).

### Video

The `VideoRecorder` can be used to record videos of the agent during training (taken from [SAC-AE](https://github.com/denisyarats/pytorch_sac_ae)). Can be used by the setting the `train.record_video` option in the corresponding .yaml-file to true.

### Evaluation

This directory includes:
- The results that were recorded while training and evaluating the agent as .csv-files.
- Several `plot_*.py` files used for generating plots from the recorded results.
- The actual plots, that were created.


## Experiment-Info

- The experiments for the raw-image and reconstruction agents were run on a Windows-PC using a NVIDIA GeForce RTX 2070 SUPER GPU.
- The experiments for forward-prediction agent were run on a Windows-PC using a NVIDIA GeForce RTX 3060 GPU.
