import numpy as np
from fastcore.foundation import L

__ops__ = ['add', 'sub', 'noop']


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def noop(x, _):
    """
    perform no operation "no-op"
    :param x: input 1
    :param _: input 2
    :return: input 1
    """
    return x


def get_op(op: str):
    return eval(op, globals())


def named_idx(x: np.ndarray, sfx: str):
    """
    return a list of string indices matching the array
    suffixed with sfx
    :param x: ndarray
    :param sfx: suffix
    :return:
    """
    idx = np.arange(0, x.shape[0]).tolist()
    return L([sfx + i.__str__() for i in idx])
