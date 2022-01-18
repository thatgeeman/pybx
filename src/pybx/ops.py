import numpy as np
from fastcore.foundation import L

__ops__ = ['add', 'sub', 'mul', 'noop']
voc_keys = ['x_min', 'y_min', 'x_max', 'y_max', 'label']


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


def make_array(x):
    if isinstance(x, dict):
        try:
            x = [x[k] for k in voc_keys]
        except TypeError:
            x = [x[k] for k in voc_keys[:-1]]
    # now dict made into a list too
    if isinstance(x, list):
        if len(x) > 4:
            return np.asarray(x[:4]), x[-1]
        else:
            return np.asarray(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise NotImplementedError


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


def intersection_box(b1: np.ndarray, b2: np.ndarray):
    """
    return the intersection box given two boxes
    :param b1:
    :param b2:
    :return:
    """
    if not isinstance(b1, np.ndarray):
        raise TypeError('expected ndarrays')
    top_edge = np.max(np.vstack([b1, b2]), axis=0)[:2]
    bot_edge = np.min(np.vstack([b1, b2]), axis=0)[2:]
    if (bot_edge > top_edge).all():
        return np.hstack([top_edge, bot_edge])
    raise NoIntersection


class NoIntersection(Exception):
    pass
