allowed_ops = 'add', 'sub', 'noop'


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


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