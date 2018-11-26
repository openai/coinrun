#!/usr/bin/env python

"""
Train an agent using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from coinrun import tb_utils, setup_utils, policies, wrappers, ppo2
import coinrun.main_utils as utils

from baselines.common import set_global_seeds

from mpi4py import MPI

import argparse

import time
from argparse import Namespace
from coinrun.config import Config

def main():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    nenvs = Config.NUM_ENVS
    total_timesteps = int(8e6 * nenvs)
    save_interval = args.save_interval

    env = utils.make_general_env(nenvs, seed=rank)

    with tf.Session(config=config) as sess:
        env = wrappers.add_final_wrappers(env)
        
        policy = policies.get_policy()

        mean_rewards = ppo2.learn(policy=policy,
                             env=env,
                             save_interval=save_interval,
                             nsteps=Config.NUM_STEPS,
                             nminibatches=Config.NUM_MINIBATCHES,
                             lam=0.95,
                             gamma=Config.GAMMA,
                             noptepochs=Config.PPO_EPOCHS,
                             log_interval=1,
                             ent_coef=Config.ENTROPY_COEFF,
                             lr=lambda f : f * Config.LEARNING_RATE,
                             cliprange=lambda f : f * 0.2,
                             total_timesteps=total_timesteps)

if __name__ == '__main__':
    main()

