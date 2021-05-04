# ViZDoom Bot
We're ViZDoom Bot team of CSCI527 Spring 2021. This repository contains
all the code and part of saved model weights of our team throughout
the whole semester. Please visit [our website](https://vizdoombot2021.netlify.app/)
for more details.

## Introduction of ViZDoom
The ViZDoom competition was an artificial intelligence competition
held from 2016 to 2018 in which artificial agents making use of
the [ViZDoom platform](http://vizdoom.cs.put.edu.pl/) and having
access to only Doom's visual buffer and UI statistics engaged in
a series of tests on separate tracks, including death matches and
navigation challenges, to determine which agent / AI method was
superior at playing the game doom.

The goal of this project will be to use current machine learning
techniques to construct an artificial agent capable of competent
play at the game Doom in general, and more specifically an agent
which would have been capable of successfully competing in the ViZDoom
competition and whose performance will (preferably) exceed that of
the best previous competitors in the competition.

## Project Structure
- archive_code: Self-implemented agents using TensorFlow v2, which are
  not used after midterm and became archived since then. Please refer to
  individual README files within each one of them.
- archive_models: Saved models for some scenarios. Copy zip files into the folder
  with the same name located in the repo root to use them.
- common: Common building blocks of our agent. The final version of our agent
  is built with the help of [stable-baselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html).
  We added some custom networks, functions, and wrappers on top of it.
- scenarios: All the scenarios used while training our agents.
- All other folders: Each of them specifies a particular setting (hyper-params,
  action space, network params, etc.) for the agent that can be then trained and
  evaluated on a particular scenario.

## Dependencies
See requirements.txt for the full list.

**WARNING**: stable-baselines only support Tensorflow <= 1.15
for now. So be sure to install the right version. In addition,
if you plan to use GPU, here's a list of cuda lib versions
for your reference:
- cuda: 10.0.130
- cudnn: 7.6.5

**TIPS**: using [Conda](https://docs.conda.io/en/latest/miniconda.html)
to manage different lib versions and virtual environments is
highly recommended.

## Directions to Run the Agent
After installing all the required dependencies, follow the directions
below to train & evaluate the agent. Take simpler_basic_lstm as an example:

```shell
cd simpler_basic_lstm

# Edit constants.py to change action space, input frame size, game args, etc.

# Edit params.py to tweak parameters and hyper-parameters.

# Run the training process. If interrupted halfway, models will be safely saved.
python3 train_ppo_lstm.py

# Run the evaluate process
python3 evaluate_ppo_lstm.py

# Record a video of the agent playing the game along with visualized output of
# the policy network.
python3 record_evaluate_ppo_lstm.py
```

## Directions to Run a Deathmatch
```shell
# In terminal 1
# Start the host
cd flatmap_host
# Edit host_simple.py to adjust number of clients, number of built-in bots,
# and whether the host will be a player or a specular. Then:
python3 host_simple.py

# In terminal 2
# Start agent 1
cd flatmap_agent
python3 deathmatch.py

# (Optional)
# In terminal 3
# Start agent 2
cd flatmap_agent2
python3 deathmatch.py
```

## Scenarios Information
'_lvn' postfix where n ranges from 1 to 9 indicates the difficulty level
of built-in bots with varied Speed and Health values. Specifically:
- lv1: Speed = 0.2, Health = 40. (Pistol only)
- lv2: Speed = 0.2, Health = 40.
- lv3: Speed = 0.4, Health = 40.
- lv4: Speed = 0.4, Health = 60.
- lv5: Speed = 0.6, Health = 60.
- lv6: Speed = 0.8, Health = 60.
- lv7: Speed = 0.8, Health = 80.
- lv8: Speed = 1.0, Health = 100.
- lv9: Speed = random(0.2, 1.0), Health = random(40, 100).

'_aug' postfix indicates the agent is trained using AugmentedPPO in which
force exploration is enabled with explore_probability = 0.1.

'_larger' postfix indicates the agent uses a larger network instead of
the default nature_cnn provided by stable-baselines.

- cig_map1: The scenario for Track 1a (limited deathmatch).
- cig_map1_less_penalty: The scenario for Track 1a with less shaping penalties for suicide.
- cig_map2/3: The scenario for Track 2 (full deathmatch).
- deadly_corridor: Refer to official docs of ViZDoom.
- deathmatch: Refer to official docs of ViZDoom.
- defend_the_center: Refer to official docs of ViZDoom.
- defend_the_line: Refer to official docs of ViZDoom.
- flatmap_agent/agent2: Join a deathmatch as a client.
- flatmap_host: Host a deathmatch as the host.
- flatmap: Simplified version of cig_map1.
- flatmap_acc: Penalize more on missing shots in hope of boosting aiming accuracy.
- health_gathering: Refer to official docs of ViZDoom.
- midterm_demo_2/3: A scenario where the agent is spawned with scattered pickups
  and with a few monsters including HellKnights. The goal is to kill all HellKnights
  within time limit.
- my_cig_01_xxxx_xxx: Deprecated version of cig_map1. Preliminary tests before midterm.
- my_way_home: Refer to official docs of ViZDoom.
- my_way_home_lstm: Agent trained on my_way_home without any curiosity mechanisms.
- my_way_home_lstm_curiosity: Agent trained on my_way_home with RDN.
- oblige_0: Agent trained and tested on the #0 of PyOblige maps.
- oblige_multi: Agent trained and tested on different PyOblige maps.
- oblige_pretrain_aiming: A scenario where the agent id spawned with a random weapon
  facing a random monster positioned at a random place. The aim of the scenario is to
  teach the agent to aim and shoot, providing a better starting point for training
  on oblige maps afterwards.
- predict_position: Refer to official docs of ViZDoom.
- rocket_basic/basic2: Refer to official docs of ViZDoom.
- rocket_basic_easy: Simpler version of rocket_basic where the monster can be killed
  in one shot.
- rocket_basic_easy_random: Randomized version of rocket_basic_easy where the
  initial weapon and monster type are chosen randomly.
- rocket_medium: Harder version of rocket_basic_easy where in some cases the agent
  must move around to pick up the rocket launcher and rockets before shooting at
  the monster.
- simpler_basic: Refer to official docs of ViZDoom.
- simpler_deathmatch: Simpler version of deathmatch with reduced action space.
- simpler_deathmatch_lstm: simpler_deathmatch with policy network integrating LSTM.
- simpler_my_way_home: Simpler version of my_way_home where only 4 rooms are in the
  map instead of 7.
- simpler_my_way_home_lstm_curiosity: simpler_my_way_home with LSTM + RDN.
- simpler_my_way_home_lstm_icm: simpler_my_way_home with LSTM + ICM.
- simpler_my_way_home_lstm_icm2: simpler_my_way_home_icm with curiosity_weight = 10.0.
- simpler_my_way_home_lstm_icm3: simpler_my_way_home_icm with curiosity_weight = 100.0.
- simpler_my_way_home_lstm_icm4: simpler_my_way_home_icm with curiosity_weight = 5.0.
- take_cover: Refer to official docs of ViZDoom.
