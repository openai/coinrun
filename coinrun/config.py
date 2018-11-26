from mpi4py import MPI
import numpy as np
import argparse
import platform

class ConfigSingle(object):
    def __init__(self):
        self.WORKDIR = '/root/data/saved_models/'
        self.LOG_ALL_MPI = True
        self.SYNC_FROM_ROOT = True

        arg_keys = []

        bool_keys = []
        type_keys = []
        array_keys = []

        bool_keys.append(('eval', 'eval'))
        bool_keys.append(('eval-all', 'eval_all'))
        bool_keys.append(('render-obs', 'render_obs'))
        bool_keys.append(('test', 'test'))
        bool_keys.append(('highd', 'high_difficulty'))
        bool_keys.append(('fullr', 'full_render'))

        type_keys.append(('ne', 'num_envs', int, 32, True))
        type_keys.append(('si', 'save_interval', int, 10))
        type_keys.append(('lstm', 'lstm_mode', int, 0, True))
        type_keys.append(('rep', 'rep', int, 1))
        type_keys.append(('ns', 'num_steps', int, 256))
        type_keys.append(('nmb', 'num_minibatches', int, 8))
        type_keys.append(('ppoeps', 'ppo_epochs', int, 3))
        type_keys.append(('arch', 'architecture', int, 1, True))

        type_keys.append(('nlev', 'num_levels', int, 0, True))
        type_keys.append(('set-seed', 'set_seed', int, 0, True))
        type_keys.append(('count', 'count', int, 7))
        type_keys.append(('pvi', 'paint_vel_info', int, -1, True))
        type_keys.append(('urb', 'use_random_blotches', int, 0))
        type_keys.append(('fs', 'frame_stack', int, 1, True))
        type_keys.append(('ubw', 'use_black_white', int, 0, True))
        type_keys.append(('norm', 'norm_mode', int, 0, True))

        type_keys.append(('ent', 'entropy_coeff', float, .01))
        type_keys.append(('lr', 'learning_rate', float, 5e-4))
        type_keys.append(('gamma', 'gamma', float, 0.999))
        type_keys.append(('l2', 'l2_weight', float, 0.0))
        type_keys.append(('dropout', 'dropout', float, 0.0, True))
        type_keys.append(('rap', 'random_action_prob', float, 0.0))

        type_keys.append(('runid', 'run_id', str, 'tmp'))
        type_keys.append(('gamet', 'game_type', str, 'standard', True)) # one of {'standard', 'platform', 'maze'}
        type_keys.append(('resid', 'restore_id', str, None))

        array_keys.append(('layer_depths', [16, 32, 32]))

        self.RES_KEYS = []

        for tk in type_keys:
            arg_keys.append(self.process_field(tk[1]))

            if (len(tk) > 4) and tk[4]:
                self.RES_KEYS.append(tk[1])

        for bk in bool_keys:
            arg_keys.append(bk[1])

            if (len(tk) > 2) and tk[2]:
                self.RES_KEYS.append(tk[1])

        for ak in array_keys:
            arg_keys.append(ak[0])
            self.RES_KEYS.append(ak[0])

        self.arg_keys = arg_keys
        self.bool_keys = bool_keys
        self.type_keys = type_keys
        self.array_keys = array_keys

        self.load_data = {}
        self.args_dict = {}

    def is_test_rank(self):
        if self.TEST:
            rank = MPI.COMM_WORLD.Get_rank()
            size = MPI.COMM_WORLD.Get_size()

            return rank % 2 == 1

        return False

    def get_test_frac(self):
        return .5 if self.TEST else 0

    def get_load_data(self, load_key='default'):
        if not load_key in self.load_data:
            return None

        return self.load_data[load_key]

    def set_load_data(self, ld, load_key='default'):
        self.load_data[load_key] = ld

    def process_field(self, name):
        return name.replace('-','_')

    def deprocess_field(self, name):
        return name.replace('_','-')

    def parse_all_args(self, args):
        for ak in self.array_keys:
            self.args_dict[ak[0]] = ak[1]
            
        args_dict = vars(args)
        self.parse_args_dict(args_dict)

    def parse_args_dict(self, update_dict):
        self.args_dict.update(update_dict)

        for ak in self.args_dict:
            val = self.args_dict[ak]

            if isinstance(val, str):
                val = self.process_field(val)

            setattr(self, ak.upper(), val)

        self.compute_args_dependencies()

    def compute_args_dependencies(self):
        if self.is_test_rank():
            self.NUM_LEVELS = 0
            self.USE_RANDOM_BLOTCHES = 0
            self.RANDOM_ACTION_PROB = 0
            self.HIGH_DIFFICULTY = 1

        if self.PAINT_VEL_INFO < 0:
            if self.GAME_TYPE == 'standard':
                self.PAINT_VEL_INFO = 1
            else:
                self.PAINT_VEL_INFO = 0

        self.TRAIN_TEST_COMM = MPI.COMM_WORLD.Split(1 if self.is_test_rank() else 0, 0)

    def get_load_filename(self, base_name=None, restore_id=None):
        if restore_id is None:
            restore_id = Config.RESTORE_ID

        if restore_id is None:
            return None
        
        comm = MPI.COMM_WORLD
        filename = Config.get_save_file_for_rank(0, self.process_field(restore_id), base_name=base_name)

        return filename

    def get_save_path(self, runid=None):
        return self.WORKDIR + self.get_save_file(runid)

    def get_save_file_for_rank(self, rank, runid=None, base_name=None):
        if runid is None:
            runid = self.RUN_ID

        extra = ''

        if base_name is not None:
            extra = '_' + base_name

        return 'sav_' + runid + extra + '_' + str(rank)

    def get_save_file(self, runid=None, base_name=None):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        return self.get_save_file_for_rank(rank, runid, base_name=base_name)

    def get_arg_text(self):
        arg_strs = []

        for key in self.args_dict:
            arg_strs.append(key + '=' + str(self.args_dict[key]))

        return arg_strs

    def get_args_dict(self):
        _args_dict = {}
        _args_dict.update(self.args_dict)

        return _args_dict
        
    def initialize_args(self, **kwargs):
        default_args = {}

        for tk in self.type_keys:
            default_args[self.process_field(tk[1])] = tk[3]

        for bk in self.bool_keys:
            default_args[bk[1]] = False

        default_args.update(kwargs)

        parser = argparse.ArgumentParser()

        for tk in self.type_keys:
            parser.add_argument('-' + tk[0], '--' + self.deprocess_field(tk[1]), type=tk[2], default=default_args[tk[1]])

        for bk in self.bool_keys:
            parser.add_argument('--' + bk[0], dest=bk[1], action='store_true')
            bk_kwargs = {bk[1]: default_args[bk[1]]}
            parser.set_defaults(**bk_kwargs)

        args = parser.parse_args()

        self.parse_all_args(args)

        return args

Config = ConfigSingle()