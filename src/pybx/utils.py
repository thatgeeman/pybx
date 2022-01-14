allowed_ops = 'add', 'subtract', 'noop'
voc_keys = ['x_min', 'y_min', 'x_max', 'y_max', 'label']


def noop(x):
    return x


def get_op(op: str):
    assert 'np' in globals(), "numpy not imported: import numpy as np"
    return eval(op) if op == 'noop' else eval(f'np.{op}')


