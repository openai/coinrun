from coinrun.config import Config

import os
import joblib

def load_for_setup_if_necessary():
    restore_file(Config.RESTORE_ID)

def restore_file(restore_id, load_key='default'):
    if restore_id is not None:
        load_file = Config.get_load_filename(restore_id=restore_id)
        filepath = file_to_path(load_file)
        load_data = joblib.load(filepath)

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

def setup_and_load(use_cmd_line_args=True, **kwargs):
    """
    Initialize the global config using command line options, defaulting to the values in `config.py`.

    `use_cmd_line_args`: set to False to ignore command line arguments passed to the program
    `**kwargs`: override the defaults from `config.py` with these values
    """
    args = Config.initialize_args(use_cmd_line_args=use_cmd_line_args, **kwargs)

    load_for_setup_if_necessary()

    return args

def file_to_path(filename):
    return os.path.join(Config.WORKDIR, filename)