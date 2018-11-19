import tensorflow as tf
import os
import joblib
import numpy as np

from mpi4py import MPI

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from coinrun.config import Config
from coinrun import setup_utils, wrappers

import platform

def make_general_env(num_env, seed=0, use_sub_proc=True):
    from coinrun import coinrunenv
    
    env = coinrunenv.make(Config.GAME_TYPE, num_env)

    if Config.FRAME_STACK > 1:
        env = VecFrameStack(env, Config.FRAME_STACK)

    epsilon = Config.EPSILON_GREEDY

    if epsilon > 0:
        env = wrappers.EpsilonGreedyWrapper(env, epsilon)

    return env

def file_to_path(filename):
    return setup_utils.file_to_path(filename)

def load_all_params(sess):
    load_params_for_scope(sess, 'model')

def load_params_for_scope(sess, scope, load_key='default'):
    load_data = Config.get_load_data(load_key)
    if load_data is None:
        return False

    params_dict = load_data['params']

    if scope in params_dict:
        print('Loading saved file for scope', scope)

        loaded_params = params_dict[scope]

        loaded_params, params = get_savable_params(loaded_params, scope, keep_heads=True)

        restore_params(sess, loaded_params, params)
    
    return True

def get_savable_params(loaded_params, scope, keep_heads=False):
    params = tf.trainable_variables(scope)
    filtered_params = []
    filtered_loaded = []

    if len(loaded_params) != len(params):
        print('param mismatch', len(loaded_params), len(params))
        assert(False)

    for p, loaded_p in zip(params, loaded_params):
        keep = True

        if any((scope + '/' + x) in p.name for x in ['v','pi']):
            keep = keep_heads

        if keep:
            filtered_params.append(p)
            filtered_loaded.append(loaded_p)
        else:
            print('drop', p)
            

    return filtered_loaded, filtered_params

def restore_params(sess, loaded_params, params):
    if len(loaded_params) != len(params):
        print('param mismatch', len(loaded_params), len(params))
        assert(False)

    restores = []
    for p, loaded_p in zip(params, loaded_params):
        print('restoring', p)
        restores.append(p.assign(loaded_p))
    sess.run(restores)

def save_params_in_scopes(sess, scopes, filename, base_dict=None):
    data_dict = {}

    if base_dict is not None:
        data_dict.update(base_dict)

    save_path = file_to_path(filename)

    data_dict['args'] = Config.get_args_dict()
    param_dict = {}

    for scope in scopes:
        params = tf.trainable_variables(scope)

        if len(params) > 0:
            print('saving scope', scope, filename)
            ps = sess.run(params)

            param_dict[scope] = ps
        
    data_dict['params'] = param_dict
    joblib.dump(data_dict, save_path)

def setup_mpi_gpus():
    if 'RCALL_NUM_GPU' not in os.environ:
        return
    num_gpus = int(os.environ['RCALL_NUM_GPU'])
    node_id = platform.node()
    nodes = MPI.COMM_WORLD.allgather(node_id)
    local_rank = len([n for n in nodes[:MPI.COMM_WORLD.Get_rank()] if n == node_id])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % num_gpus)

def is_mpi_root():
    return MPI.COMM_WORLD.Get_rank() == 0

def tf_initialize(sess):
    sess.run(tf.initialize_all_variables())
    sync_from_root(sess)
    
def sync_from_root(sess, vars=None):
    if vars is None:
        vars = tf.trainable_variables()

    if Config.SYNC_FROM_ROOT:
        rank = MPI.COMM_WORLD.Get_rank()
        print('sync from root', rank)
        for var in vars:
            if rank == 0:
                MPI.COMM_WORLD.bcast(sess.run(var))
            else:
                sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))

def mpi_average(values):
    return mpi_average_comm(values, MPI.COMM_WORLD)

def mpi_average_comm(values, comm):
    size = comm.size

    x = np.array(values)
    buf = np.zeros_like(x)
    comm.Allreduce(x, buf, op=MPI.SUM)
    buf = buf / size

    return buf

def mpi_average_train_test(values):
    return mpi_average_comm(values, Config.TRAIN_TEST_COMM)
    
def mpi_print(*args):
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print(*args)

def process_ep_buf(epinfobuf, tb_writer=None, suffix='', step=0):
    rewards = [epinfo['r'] for epinfo in epinfobuf]
    rew_mean = np.nanmean(rewards)

    if Config.SYNC_FROM_ROOT:
        rew_mean = mpi_average_train_test([rew_mean])[0]

    if tb_writer is not None:
        tb_writer.log_scalar(rew_mean, 'rew_mean' + suffix, step)

    aux_dicts = []

    if len(epinfobuf) > 0 and 'aux_dict' in epinfobuf[0]:
        aux_dicts = [epinfo['aux_dict'] for epinfo in epinfobuf]

    if len(aux_dicts) > 0:
        keys = aux_dicts[0].keys()

        for key in keys:
            sub_rews = [aux_dict[key] for aux_dict in aux_dicts]
            sub_rew = np.nanmean(sub_rews)

            if tb_writer is not None:
                tb_writer.log_scalar(sub_rew, key, step)

    return rew_mean
