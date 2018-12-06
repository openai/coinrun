import numpy as np
from coinrun import setup_utils, make

def test_coinrun():
    setup_utils.setup_and_load(use_cmd_line_args=False)
    env = make('standard', num_envs=16)
    for _ in range(100):
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _obs, _rews, _dones, _infos = env.step(acts)
    env.close()


if __name__ == '__main__':
    test_coinrun()