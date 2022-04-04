import inspect

import numpy as np
from fastcore.foundation import L

from .excepts import NoIntersection

__ops__ = ['add', 'sub', 'mul', 'noop']
voc_keys = ['x_min', 'y_min', 'x_max', 'y_max', 'label']
label_keys = ['label', 'class_name', 'class', 'name', 'class_id', 'object', 'item']


def add(x, y):
    """Add two objects."""
    return x + y


def sub(x, y):
    """Subtract two objects."""
    return x - y


def mul(x, y):
    """Multiply two objects."""
    return x * y


def noop(x, _):
    """Perform no operation ("no-op") on `x`.
    :param x: input object 1
    :param _: input object 2
    :return: input object 1
    """
    return x


def get_op(op: str):
    """Given a string of aps.__ops__, return the function reference."""
    return eval(op, globals())


def make_array(x):
    """Method to convert a single dict or a list to an array.
    :param x: dict with keys {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "label": 'none'}
    :return: `coords` as `ndarray`, `label` as `list`
    """
    if isinstance(x, dict):
        # dict into list
        try:
            x = [x[k] for k in voc_keys]
        except TypeError:
            x = [x[k] for k in voc_keys[:-1]]

    if isinstance(x, tuple):
        x = list(x)

    if isinstance(x, (list, np.ndarray)) and len(x) >= 4:
        # lists of a single list would fail this check
        if len(x) > 4:
            return np.asarray(x[:4]), [x[-1]]
        else:
            return np.asarray(x)
    else:
        raise NotImplementedError(f'{inspect.stack()[0][3]} of {__name__}: Expected {dict} got {type(x)}.')


def named_idx(x: np.ndarray, sfx: str = ''):
    """Return a list of indices as `str` matching the array size, suffixed with `sfx`.
    :param x: 1-dimensional array
    :param sfx: suffix to be added to the index
    :return: list of strings
    """
    idx = np.arange(0, x.shape[0]).tolist()
    return L([sfx + i.__str__() for i in idx])


def intersection_box(b1: np.ndarray, b2: np.ndarray):
    """Return the box that intersects two boxes in `pascal_voc` format."""
    if not isinstance(b1, np.ndarray):
        raise TypeError(f'{inspect.stack()[0][3]} of {__name__}: Expected ndarrays.')
    top_edge = np.max(np.vstack([b1, b2]), axis=0)[:2]
    bot_edge = np.min(np.vstack([b1, b2]), axis=0)[2:]
    if (bot_edge > top_edge).all():
        return np.hstack([top_edge, bot_edge])
    raise NoIntersection


def update_keys(annots: dict, default_keys=None):
    """Modify the default class `label` key that the `JsonBx` method looks for.
    By default, `JsonBx` uses the parameter `ops.voc_keys` and looks for the
    key "label" in the dict. If called, `update_keys` looks inside the parameter
    `ops.label_keys` for matching key in the passed `annots` and uses
    this as the key to identify class label. Fixes #3.
    :param annots: dictionary of annotations
    :param default_keys: `voc_keys` by default
    :return: new keys with updated label key
    """
    if default_keys is None:
        default_keys = voc_keys
    label_key = None
    for k, v in annots.items():
        if k in label_keys:
            label_key = k
            break
    return default_keys[:-1] + [label_key] if label_key is not None else default_keys
