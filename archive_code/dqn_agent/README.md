# DQN agent for VizDoom
This repo consists of various DQN agents trained
for different scenarios of VizDoom.

## Project Structure
- dqn_common: Common classes and functions for
  DQN, including DQNAgent, GameWrapper, ReplayBuffer,
  and process_frame
- scenarios: scenario files of VizDoom
- simpler_basic: DQN agent trained for simpler_basic.cfg
- deadly_corridor: DQN agent trained for deadly_corridor.cfg

## Training & Evaluation
To train an agent for a specific scenario, cd into corresponding
folder and run
```shell
python3 train.py
```

Similarly, to evaluate a trained agent, cd into that folder and
run
```shell
python3 evaluate.py
```

For each agent, you can tweak the model architecture in model.py
and parameters/hyper-parameters in params.py.
