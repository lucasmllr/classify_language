import os
from os.path import join, exists
import yaml
from dotmap import DotMap
from shutil import copytree, copy, rmtree
from sys import exit


def load_config(path):
    with open(path) as f:
        params = yaml.load(f)
    return DotMap(params)
    

def init_experiment_from_config(path, check_overriding=True):
    """loads a config file at path, initializes a directory for the experiment,
    copies code and config file.
    It is optionally checked whether the experiment already exists and whether it should
    be deleted.
    Args:
        path (PathLike): path to config yml file
        check_overriding (bool): wheter to check if the defined experiment dir already exists
    """
    # load parameters
    with open(path) as f:
        params = yaml.load(f)
    params = DotMap(params)
    # check overriding
    exp_path = join(params.saving.root, params.saving.name)
    if exists(exp_path):
        if check_overriding:
            inpt = input(f'{exp_path} exists, do you want to override it? ')
            if not inpt == 'y':
                print('exiting')
                exit()
        print(f'overriding {exp_path}')
        rmtree(exp_path, ignore_errors=True)
    # make dir and copy code
    os.makedirs(exp_path)
    code_path = join(exp_path, 'code')
    copytree('./src', code_path)
    copy(path, join(exp_path, 'config.yml'))
    return params