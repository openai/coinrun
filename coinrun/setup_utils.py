from coinrun.config import Config

from mpi4py import MPI
import os
import joblib

def load_for_setup_if_necessary():
    load_file = Config.get_load_filename()

    restore_file(Config.RESTORE_ID)

def get_load_data(restore_id=None):
    load_file = Config.get_load_filename(restore_id=restore_id)

    filepath = file_to_path(load_file)
    return joblib.load(filepath)

def restore_file(restore_id, load_key='default'):
    if restore_id is not None:
        load_data = get_load_data(restore_id)

        Config.set_load_data(load_data, load_key=load_key)

        restored_args = load_data['args']
        sub_dict = {}
        res_keys = Config.RES_KEYS

        for key in res_keys:
            if key in restored_args:
                sub_dict[key] = restored_args[key]
            else:
                print('warning key %s not restored' % key)

        Config.parse_args_dict(sub_dict)
    
    from coinrun.coinrunenv import init_args_and_threads
    init_args_and_threads(4)

def setup_and_load_with_args(args):
    Config.parse_all_args(args)

    load_for_setup_if_necessary()

def setup_and_load(**kwargs):
    args = Config.initialize_args(**kwargs)

    load_for_setup_if_necessary()

    return args

def file_to_path(filename):
    return os.path.join(Config.WORKDIR, filename)