# Stable Baselines Agent
This repo contains some RL agents trained with 
[stable-baselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html).
Main purpose of this repo is to provide a reference performance
on multiple RL algorithms that has been tested widely to work.

## Requirements
See requirements.txt for the full list.

**WARNING**: stable-baselines only support Tensorflow <= 1.15
for now. So be sure to install the right version. In addition,
if you plan to use GPU, here's a list of cuda lib versions
for your reference:
- cuda: 10.0.130
- cudnn: 7.6.5

**TIPS**: using [Conda](https://docs.conda.io/en/latest/miniconda.html)
to manage different lib versions and virtual environments are
highly recommended.
