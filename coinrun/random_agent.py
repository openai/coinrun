import numpy as np
from coinrun import setup_utils, make


def random_agent(num_envs=1, max_steps=100000):
    setup_utils.setup_and_load(use_cmd_line_args=False)
    env = make('standard', num_envs=num_envs)
    for step in range(max_steps):
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _obs, rews, _dones, _infos = env.step(acts)
        print("step", step, "rews", rews)
    env.close()


if __name__ == '__main__':
    random_agent()