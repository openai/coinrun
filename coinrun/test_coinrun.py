import numpy as np
from coinrun import setup_utils, make

def test_coinrun():
    setup_utils.setup_and_load(cmd_line_args=[])
    env = make('CoinRun-v0', num_envs=16)
    for _ in range(1000):
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _obs, _rews, _dones, _infos = env.step(acts)
    env.close()


if __name__ == '__main__':
    test_coinrun()