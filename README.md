**Status:** Archive (code is provided as-is, no updates expected)

# Quantifying Generalization in Reinforcement Learning

#### [[Blog Post]](https://blog.openai.com/?????) [[Paper]](https://arxiv.org/abs/???????)

This is code for the environments and training used in the paper [Quantifying Generalization in Reinforcement Learning](https://arxiv.org/abs/???????).

Authors: Karl Cobbe, ???.

## Install

You should install the package in development mode so you can easily change the files.  You may also want to create a virtualenv before installing since it depends on a specific version of OpenAI baselines.

```
# Linux
apt-get install qtbase5-dev mpich
# Mac
brew install qt open-mpi

git clone https://github.com/openai/coinrun.git
cd coinrun
pip install tensorflow  # or tensorflow-gpu
pip install -r requirements.txt
pip install -e .
```

Note that this does not compile the environment, the environment will be compiled when the `coinrun` package is imported.

## Try it out

Try the environment out with the keyboard:

```
python -m coinrun.interactive
```

Train an agent using PPO:

```
python -m coinrun.train_agent --run-id myrun --save-interval 1
```

After each parameter update, this will save a copy of the agent to `./saved_models/`.

Run parallel training using MPI:

```
mpirun -np 8 python -m coinrun.train_agent --run-id myrun
```

View training options:

```
python -m coinrun.train_agent --help
```

Watch a trained agent play a level:

```
python -m coinrun.enjoy --restore-id myrun --fullr
```

Example random agent script:

```
import numpy as np
from coinrun import setup_utils, make

setup_utils.setup_and_load(cmd_line_args=[])
env = make('CoinRun-v0', num_envs=16)
for _ in range(100):
    acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    _obs, _rews, _dones, _infos = env.step(acts)
env.close()
```