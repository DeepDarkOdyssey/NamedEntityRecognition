"""This module implements some utility functions"""

import json
import logging
import os
from collections import namedtuple


def load_config(dict_or_ntuple, json_path=None, config_dict=None):
    # TODO: maybe can use the bunch package instead of namedtuple
    if type(dict_or_ntuple) is dict:
        d = dict_or_ntuple
    else:
        try:
            d = dict_or_ntuple._asdict()
        except AttributeError:
            raise TypeError('dict_or_ntuple must be either a dict or a namedtuple')

    if json_path:
        with open(json_path) as f:
            some_config = json.load(f)
        d.update(some_config)

    if config_dict:
        d.update(config_dict)

    return namedtuple('Config', d.keys())(**d)


def set_logger(config):
    logger = logging.getLogger(config.logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(config.exp_dir, 'log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)


def prepare_dirs(config):
    """Make sure every needed directories exists, and update the config"""

    if not os.path.exists(config.experiments_dir):
        os.mkdir(config.experiments_dir)
    if not os.path.exists(os.path.join(config.experiments_dir, config.model_name)):
        os.mkdir(os.path.join(config.experiments_dir, config.model_name))
    if not os.path.exists(os.path.join(config.experiments_dir, config.model_name, config.exp_name)):
        os.mkdir(os.path.join(config.experiments_dir, config.model_name, config.exp_name))

    model_dir = os.path.join(config.experiments_dir, config.model_name)
    exp_dir = os.path.join(config.experiments_dir, config.model_name, config.exp_name)
    sub_dir_names = ['summaries', 'checkpoints', 'results']
    for dir_name in sub_dir_names:
        if not os.path.exists(os.path.join(exp_dir, dir_name)):
            os.mkdir(os.path.join(exp_dir, dir_name))
    config_dict = {
        'model_dir': model_dir,
        'exp_dir': exp_dir,
        'summary_dir': os.path.join(exp_dir, 'summaries'),
        'ckpt_dir': os.path.join(exp_dir, 'checkpoints'),
        'result_dir': os.path.join(exp_dir, 'results')
    }
    config = load_config(config, config_dict=config_dict)

    return config
