'''
Do NOT move this file, otherwise get_project_root doesnt work anymore. 
'''


import os
from pathlib import Path
from typing import List
import json
from pathlib import Path
from itertools import repeat
from collections.abc import Mapping
from collections import OrderedDict
from time import time
import numpy as np
import os


def tictoc():
    """
    Returns time in seconds since the last time the function was called.
    For the initial call 0 is returned.
    """
    if not hasattr(tictoc, 'tic'):
        tictoc.tic = time()

    toc = time()
    dt = toc - tictoc.tic
    tictoc.tic = toc

    return dt


def ptictoc(name=None):
    """
    Returns and prints (!) time in seconds since the last time the function was called.
    For the initial call 0 is returned.
    """
    if not hasattr(tictoc, 'tic'):
        tictoc.tic = time()
        initial = True
    else:
        initial = False

    toc = time()
    dt = toc - tictoc.tic
    tictoc.tic = toc

    if not initial:
        if name is None:
            print('dt = {}'.format(dt))
        else:
            print('dt_{} = {}'.format(name, dt))

    return dt

# Define a context manager to suppress stdout and stderr.
# Taken from: https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def read_config_json(fname):
    """
    Load config from json file and perform consistency checks
    Performs consistency checks of the config file for the reformer model and flow data.
    """

    fname = Path(fname)
    with fname.open('rt') as handle:
        config = json.load(handle, object_hook=OrderedDict)

    # Check validity of config
    c = config['arch']['args']
    c_model_name = config['arch']['type']
    if c_model_name == 'FlowTransformer':
        # c_data = config['data_loader']['args']
        # assert c['axial_pos_shape'][0]*c['axial_pos_shape'][1] == c_data['sequence_length'], \
        #     'sequence_length must be axial_pos_shape_list**2'
        assert len(c['conv1d_decoder']) == len(c['conv1d_kernel']), \
            'length of decoder parameters must be equal to length of kernel parameters'

    elif c_model_name == 'UMAPHDBSCANControlModel':
        if 'multi_class_gates' in config['data_loader']['args']:
            config['trainer']['_multi_class_gates'] = config['data_loader']['args']['multi_class_gates']
        else:
            config['trainer']['_multi_class_gates'] = None

    markers = config['data_loader']['args']['markers'].replace(
        ' ', '').split(',')
    config['arch']['args']['_num_markers'] = len(markers)
    config['arch']['args']['_sequence_length'] = config['data_loader']['args']['sequence_length']


    return config


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = {key: [] for key in keys}
        self.reset()

    def reset(self):
        for key in self._data.keys():
            self._data[key] = []
    def update_key(self, key):
        if key not in self._data:
            self._data[key] = []

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data[key] += [value] * n # n corresponds to batchsize

    def avg(self, key):
        return np.mean(self._data[key])

    def median(self, key):
        return np.median(self._data[key])

    def data(self):
        return self._data

    def get_metric_names(self) -> List[str]:
        return list(self._data.keys())

    def result(self):
        avg_dict = {key: self.avg(key) for key in self._data.keys()}
        median_dict = {key: self.median(key) for key in self._data.keys()}
        return avg_dict, median_dict


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def dict_merge(dct, merge_dct, verify=False):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :param verify: checks if no entry is added to the dictionary
    :return: None
    """
    #     dct = copy.copy(dct)
    changes_values = {}
    changes_lists = {}

    for k, _ in merge_dct.items():
        if verify:
            assert k in dct, 'key "{}" is not part of the default dict'.format(
                k)
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            changes_lists[k] = dict_merge(dct[k], merge_dct[k], verify=verify)
        else:
            if k in dct and dct[k] != merge_dct[k]:
                changes_values[k] = merge_dct[k]

            dct[k] = merge_dct[k]

    _sorted = []
    for k, _ in dct.items():
        if k in changes_values:
            _sorted.append((k, changes_values[k]))
        elif k in changes_lists:
            _sorted.extend(changes_lists[k])

    return _sorted


