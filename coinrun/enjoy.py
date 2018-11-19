"""
Load an agent trained with train_agent.py and 
"""

import time

import tensorflow as tf
import numpy as np
from coinrun import setup_utils
import coinrun.main_utils as utils
from coinrun.config import Config
from coinrun import policies, wrappers

mpi_print = utils.mpi_print

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def enjoy_env_sess(sess):
    should_render = True
    should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP

    if should_eval:
        env = utils.make_general_env(Config.NUM_EVAL)
        should_render = False
    else:
        env = utils.make_general_env(1)

    env = wrappers.add_final_wrappers(env)

    if should_render:
        from gym.envs.classic_control import rendering

    nenvs = env.num_envs

    agent = create_act_model(sess, env, nenvs)

    sess.run(tf.global_variables_initializer())
    loaded_params = utils.load_params_for_scope(sess, 'model')

    if not loaded_params:
        print('NO SAVED PARAMS LOADED')

    obs = env.reset()
    t_step = 0

    if should_render:
        viewer = rendering.SimpleImageViewer()

    should_render_obs = not Config.IS_HIGH_RES

    def maybe_render(info=None):
        if should_render and not should_render_obs:
            env.render()

    maybe_render()

    scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs

        return True

    state = agent.initial_state
    done = np.zeros(nenvs)

    while should_continue():
        action, values, state, _ = agent.step(obs, state, done)
        obs, rew, done, info = env.step(action)

        if should_render and should_render_obs:
            if np.shape(obs)[-1] % 3 == 0:
                ob_frame = obs[0,:,:,-3:]
            else:
                ob_frame = obs[0,:,:,-1]
                ob_frame = np.stack([ob_frame] * 3, axis=2)
            viewer.imshow(ob_frame)

        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1

                    if 'episode' in info[i]:
                        scores[i] += info[i].get('episode')['r']

        if t_step % 100 == 0:
            mpi_print('t', t_step, values[0], done[0], rew[0], curr_rews[0], np.shape(obs))

        maybe_render(info[0])

        t_step += 1

        if should_render:
            time.sleep(.02)

        if done[0]:
            if should_render:
                mpi_print('ep_rew', curr_rews)

            curr_rews[:] = 0

    result = 0

    if should_eval:
        mean_score = np.mean(scores) / rep_count
        max_idx = np.argmax(scores)
        mpi_print('scores', scores / rep_count)
        print('mean_score', mean_score)
        mpi_print('max idx', max_idx)

        mpi_mean_score = utils.mpi_average([mean_score])
        mpi_print('mpi_mean', mpi_mean_score)

        result = mean_score

    return result

def main():
    utils.setup_mpi_gpus()
    setup_utils.setup_and_load()
    with tf.Session() as sess:
        enjoy_env_sess(sess)

if __name__ == '__main__':
    main()