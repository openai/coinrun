"""
Python interface to the CoinRun shared library using ctypes.

On import, this will attempt to build the shared library.
"""

import os
import atexit
import random
import sys
from ctypes import c_int, c_char_p, c_float, c_bool

import gym
import gym.spaces
import numpy as np
import numpy.ctypeslib as npct
from baselines.common.vec_env import VecEnv
from baselines import logger

from coinrun.config import Config

from mpi4py import MPI
from baselines.common import mpi_util

# if the environment is crashing, try using the debug build to get
# a readable stack trace
DEBUG = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

game_versions = {
    'standard':   1000,
    'platform': 1001,
    'maze': 1002,
}

def build():
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

if DEBUG:
    lib_path = '.build-debug/coinrun_cpp_d'
else:
    lib_path = '.build-release/coinrun_cpp'

lib = npct.load_library(lib_path, os.path.dirname(__file__))
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
                          monitor_csv_policy='all',
                          rand_seed=None):
    """
    Perform one-time global init for the CoinRun library.  This must be called
    before creating an instance of CoinRunVecEnv.  You should not
    call this multiple times from the same process.
    """
    os.environ['COINRUN_RESOURCES_PATH'] = os.path.join(SCRIPT_DIR, 'assets')
    is_high_difficulty = Config.HIGH_DIFFICULTY

    if rand_seed is None:
        rand_seed = random.SystemRandom().randint(0, 1000000000)

        # ensure different MPI processes get different seeds (just in case SystemRandom implementation is poor)
        mpi_rank, mpi_size = mpi_util.get_local_rank_size(MPI.COMM_WORLD)
        rand_seed = rand_seed - rand_seed % mpi_size + mpi_rank

    int_args = np.array([int(is_high_difficulty), Config.NUM_LEVELS, int(Config.PAINT_VEL_INFO), Config.USE_DATA_AUGMENTATION, game_versions[Config.GAME_TYPE], Config.SET_SEED, rand_seed]).astype(np.int32)

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
    lib.coinrun_shutdown()

class CoinRunVecEnv(VecEnv):
    """
    This is the CoinRun VecEnv, all CoinRun environments are just instances
    of this class with different values for `game_type`

    `game_type`: int game type corresponding to the game type to create, see `enum GameType` in `coinrun.cpp`
    `num_envs`: number of environments to create in this VecEnv
    `lump_n`: only used when the environment creates `monitor.csv` files
    `default_zoom`: controls how much of the level the agent can see
    """
    def __init__(self, game_type, num_envs, lump_n=0, default_zoom=5.0):
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))

        self.NUM_ACTIONS = lib.get_NUM_ACTIONS()
        self.RES_W       = lib.get_RES_W()
        self.RES_H       = lib.get_RES_H()
        self.VIDEORES    = lib.get_VIDEORES()

        self.buf_rew = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=np.bool)
        self.buf_rgb   = np.zeros([num_envs, self.RES_H, self.RES_W, 3], dtype=np.uint8)
        self.hires_render = Config.IS_HIGH_RES
        if self.hires_render:
            self.buf_render_rgb = np.zeros([num_envs, self.VIDEORES, self.VIDEORES, 3], dtype=np.uint8)
        else:
            self.buf_render_rgb = np.zeros([1, 1, 1, 1], dtype=np.uint8)

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
            self.hires_render,
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

def make(env_id, num_envs, **kwargs):
    assert env_id in game_versions, 'cannot find environment "%s", maybe you mean one of %s' % (env_id, list(game_versions.keys()))
    return CoinRunVecEnv(env_id, num_envs, **kwargs)
