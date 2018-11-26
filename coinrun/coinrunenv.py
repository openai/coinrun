import os
import atexit
import random
import sys
from ctypes import c_int, c_char_p, c_float, c_bool

import gym
import numpy as np
import numpy.ctypeslib as npct
from baselines.common.vec_env import VecEnv
from baselines import logger
from envs import assets

from coinrun.config import Config
from coinrun import setup_utils

game_versions = {
    'CoinRun-v0':   1000,
    'CoinRunPlatforms-v0': 1001,
    'CoinRunPlatformsEasy-v0': 1002,
    'CoinRunMaze-v0': 1003,
    }

def build():
    from mpi4py import MPI
    from rl_common import mpi_util
    lrank, _lsize = mpi_util.get_local_rank_size(MPI.COMM_WORLD)
    if lrank == 0:
        dirname = os.path.dirname(__file__)
        if len(dirname):
            make_cmd = "QT_SELECT=5 make -C %s" % dirname
        else:
            make_cmd = "QT_SELECT=5 make"

        r = os.system(make_cmd)
        if r != 0:
            logger.error('coinrun: make failed')
            sys.exit(1)
    MPI.COMM_WORLD.barrier()

build()

#lib = npct.load_library('.build-debug/coinrun_cpp_d', os.path.dirname(__file__))
lib = npct.load_library('.build-release/coinrun_cpp', os.path.dirname(__file__))
lib.init.argtypes = [c_int]
lib.get_NUM_ACTIONS.restype = c_int
lib.get_RES_W.restype = c_int
lib.get_RES_H.restype = c_int
lib.get_VIDEORES.restype = c_int

lib.vec_create.argtypes = [
    c_int,    # game_type
    c_int,    # nenvs
    c_int,    # lump_n
    c_bool,   # want_hires_render
    c_float,  # default_zoom
    ]
lib.vec_create.restype = c_int

lib.vec_close.argtypes = [c_int]

lib.vec_step_async_discrete.argtypes = [c_int, npct.ndpointer(dtype=np.int32, ndim=1)]

lib.initialize_args.argtypes = [npct.ndpointer(dtype=np.int32, ndim=1)]
lib.initialize_set_monitor_dir.argtypes = [c_char_p, c_int]

lib.vec_wait.argtypes = [
    c_int,
    npct.ndpointer(dtype=np.uint8, ndim=4),    # normal rgb
    npct.ndpointer(dtype=np.uint8, ndim=4),    # larger rgb for render()
    npct.ndpointer(dtype=np.float32, ndim=1),  # rew
    npct.ndpointer(dtype=np.bool, ndim=1),     # done
    ]

already_inited = False

def init_args_and_threads(cpu_count=4,
                          monitor_csv_policy='first_env',
                          rand_seed=None):
    os.environ['COINRUN_RESOURCES_PATH'] = assets.find('gs://agi-data/assets/coinrun2/v4')
    is_high_difficulty = Config.HIGH_DIFFICULTY

    if rand_seed is None:
        rand_seed = random.SystemRandom().randint(0, 1000000000)

    level_seed_start = -1
    level_seed_end = -1

    if (Config.NUM_LEVELS > 0):
        level_seed_start = 0
        level_seed_end = Config.NUM_LEVELS - 1

    int_args = np.array([int(is_high_difficulty), level_seed_start, level_seed_end, int(Config.PAINT_VEL_INFO), Config.USE_RANDOM_BLOTCHES, rand_seed]).astype(np.int32)

    lib.initialize_args(int_args)
    lib.initialize_set_monitor_dir(logger.get_dir().encode('utf-8'), {'off': 0, 'first_env': 1, 'all': 2}[monitor_csv_policy])

    global already_inited
    if already_inited:
        return

    lib.init(cpu_count)
    already_inited = True

@atexit.register
def shutdown():
    global already_inited
    if not already_inited:
        return
    lib.shutdown()

class CoinRunVecEnv(VecEnv):
    def __init__(self, game_type, num_envs, lump_n=0, default_zoom=4.0, want_hires_render=False):
        default_zoom = 5.0
        want_hires_render = Config.FULL_RENDER

        self.NUM_ACTIONS = lib.get_NUM_ACTIONS()
        self.RES_W       = lib.get_RES_W()
        self.RES_H       = lib.get_RES_H()
        self.VIDEORES    = lib.get_VIDEORES()

        self.buf_rew = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=np.bool)
        self.buf_rgb   = np.zeros([num_envs, self.RES_H, self.RES_W, 3], dtype=np.uint8)
        if want_hires_render:
            self.buf_render_rgb = np.zeros([num_envs, self.VIDEORES, self.VIDEORES, 3], dtype=np.uint8)
        else:
            self.buf_render_rgb = np.zeros([1, 1, 1, 1], dtype=np.uint8)
        self.hires_render = want_hires_render

        num_channels = 1 if Config.USE_BLACK_WHITE else 3
        obs_space = gym.spaces.Box(0, 255, shape=[self.RES_H, self.RES_W, num_channels], dtype=np.uint8)

        super().__init__(
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=gym.spaces.Discrete(self.NUM_ACTIONS),
            )
        self.handle = lib.vec_create(
            game_versions[game_type],
            self.num_envs,
            lump_n,
            want_hires_render,
            default_zoom)
        self.dummy_info = [{} for _ in range(num_envs)]

    def __del__(self):
        if hasattr(self, 'handle'):
            lib.vec_close(self.handle)
        self.handle = 0

    def close(self):
        lib.vec_close(self.handle)
        self.handle = 0

    def reset(self):
        print("CoinRun ignores resets")
        obs, _, _, _ = self.step_wait()
        return obs

    def get_images(self):
        if self.hires_render:
            return self.buf_render_rgb
        else:
            return self.buf_rgb

    def step_async(self, actions):
        assert actions.dtype in [np.int32, np.int64]
        actions = actions.astype(np.int32)
        lib.vec_step_async_discrete(self.handle, actions)

    def step_wait(self):
        self.buf_rew = np.zeros_like(self.buf_rew)
        self.buf_done = np.zeros_like(self.buf_done)

        lib.vec_wait(
            self.handle,
            self.buf_rgb,
            self.buf_render_rgb,
            self.buf_rew,
            self.buf_done)

        obs_frames = self.buf_rgb

        if Config.USE_BLACK_WHITE:
            obs_frames = np.mean(obs_frames, axis=-1).astype(np.uint8)[...,None]
        
        return obs_frames, self.buf_rew, self.buf_done, self.dummy_info

class RandomActionEnv(gym.Wrapper):
    def __init__(self, env, prob=0.05):
        gym.Wrapper.__init__(self, env)
        self.prob = prob
        self.num_envs = env.num_envs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform()<self.prob:
            action = np.random.randint(self.env.action_space.n, size=self.num_envs)
        
        return self.env.step(action)

def make(env_id, num_envs, **kwargs):
    assert env_id in game_versions, 'cannot find environment "%s", maybe you mean one of %s' % (env_id, list(game_versions.keys()))
    return CoinRunVecEnv(env_id, num_envs, **kwargs)

def test():
    # init_args_and_threads(cpu_count=1)
    setup_utils.setup_and_load(game_name='coinrun')
    mode = 0

    if mode == 0:
        print("""Control with arrow keys,
F1, F2 -- switch resolution,
F5, F6, F7, F8 -- zoom,
F9  -- switch reconstruction target picture,
F10 -- switch lasers
        """)
        lib.test_main_loop()

    elif mode == 1:
        for j in range(10000):
            if j % 100 == 0:
                print('testing', j)
            nenvs = 16
            env = make('CoinRun-v0', nenvs)
            for _ in range(2):
                a = np.random.randint(env.action_space.n, size=(nenvs,), dtype=np.int32)
                env.step_async(a)
                _obs, _rew, _done, _info = env.step_wait()
            env.close()

if __name__ == "__main__":
    test()
