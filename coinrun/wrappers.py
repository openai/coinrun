import gym
import numpy as np

class EpsilonGreedyWrapper(gym.Wrapper):
    """
    Wrapper to perform a random action each step instead of the requested action, 
    with the provided probability.
    """
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


class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {'render.modes': []}
        env.reward_range = (-float('inf'), float('inf'))
        nenvs = env.num_envs
        self.num_envs = nenvs
        super(EpisodeRewardWrapper, self).__init__(env)

        self.aux_rewards = None
        self.num_aux_rews = None

        def reset(**kwargs):
            self.rewards = np.zeros(nenvs)
            self.lengths = np.zeros(nenvs)
            self.aux_rewards = None
            self.long_aux_rewards = None

            return self.env.reset(**kwargs)

        def step(action):
            obs, rew, done, infos = self.env.step(action)

            if self.aux_rewards is None:
                info = infos[0]
                if 'aux_rew' in info:
                    self.num_aux_rews = len(infos[0]['aux_rew'])
                else:
                    self.num_aux_rews = 0

                self.aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)
                self.long_aux_rewards = np.zeros((nenvs, self.num_aux_rews), dtype=np.float32)

            self.rewards += rew
            self.lengths += 1

            use_aux = self.num_aux_rews > 0

            if use_aux:
                for i, info in enumerate(infos):
                    self.aux_rewards[i,:] += info['aux_rew']
                    self.long_aux_rewards[i,:] += info['aux_rew']

            for i, d in enumerate(done):
                if d:
                    epinfo = {'r': round(self.rewards[i], 6), 'l': self.lengths[i], 't': 0}
                    aux_dict = {}

                    for nr in range(self.num_aux_rews):
                        aux_dict['aux_' + str(nr)] = self.aux_rewards[i,nr]

                    if 'ale.lives' in infos[i]:
                        game_over_rew = np.nan

                        is_game_over = infos[i]['ale.lives'] == 0

                        if is_game_over:
                            game_over_rew = self.long_aux_rewards[i,0]
                            self.long_aux_rewards[i,:] = 0

                        aux_dict['game_over_rew'] = game_over_rew

                    epinfo['aux_dict'] = aux_dict

                    infos[i]['episode'] = epinfo

                    self.rewards[i] = 0
                    self.lengths[i] = 0
                    self.aux_rewards[i,:] = 0

            return obs, rew, done, infos

        self.reset = reset
        self.step = step

def add_final_wrappers(env):
    env = EpisodeRewardWrapper(env)

    return env