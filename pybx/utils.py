# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_utils.ipynb.

# %% auto 0
__all__ = ['get_edges', 'validate_boxes', 'as_tuple']

# %% ../nbs/02_utils.ipynb 2
import inspect

import numpy as np

from .ops import __ops__, get_op
from .basics import get_bx

from fastcore.foundation import L

# %% ../nbs/02_utils.ipynb 3
def get_edges(image_sz: tuple, feature_sz: tuple, op="noop"):
    """Generate offsetted top `(x_min, y_min)` or bottom edges `(x_max, y_max)`
    coordinates of a given feature size based on op.
    if `op` is `noop`, gets the top edges.
    if `op` is `add`, gets the bottom edges.
    :param op: operation for calculating edges, either 'add' 'sub' 'noop'
    :param image_sz: tuple of `(W, H)` of an image
    :param feature_sz: tuple of `(W, H)` of a channel
    :return: offsetted edges of each feature
    """
    assert (
        len(image_sz) == 2
    ), f"{inspect.stack()[0][3]} of {__name__}: Expected image_sz of len 2, got {len(image_sz)}"

    assert (
        op in __ops__
    ), f"{inspect.stack()[0][3]} of {__name__}: Operator not in allowed operations: {__ops__}"
    w, h = image_sz
    nx, ny = feature_sz
    diag_edge_ofs = w / nx, h / ny
    op_ = get_op(op)
    x_ = op_(np.linspace(0, w, nx + 1), diag_edge_ofs[0])
    y_ = op_(np.linspace(0, h, ny + 1), diag_edge_ofs[1])
    mesh = np.meshgrid(x_, y_)
    edges = np.stack([m.flatten() for m in mesh], axis=-1)
    return edges


def validate_boxes(coords, image_sz, feature_sz, clip=True, min_visibility=0.25):
    """Validate calculated anchor box coords.
    :param coords: anchor box coordinates
    :param image_sz: tuple of (width, height) of an image
    :param feature_sz: tuple of (width, height) of a channel
    :param clip: whether to apply np.clip
    :param min_visibility: minimum visibility dictates the condition for a box to be considered valid. The value corresponds to the
    ratio of expected area to the calculated area after clipping to image dimensions.
    :return: anchor box coordinates in [pascal_voc] format
    """
    _max = max(image_sz[0], image_sz[1])
    # clip the boxes to image dimensions
    bxs = get_bx(coords.clip(0, _max)) if clip else get_bx(coords)
    # check if the area of the bounding box is fitting the minimum area criterion
    min_area = (
        (image_sz[0] / feature_sz[0]) * (image_sz[1] / feature_sz[1]) * min_visibility
    )
    bxs = L(list(b._coords) for b in bxs if b.area > min_area)
    return bxs


# %% ../nbs/02_utils.ipynb 7
def as_tuple(x):
    """Get x as a tuple (x, x) if not already a tuple.

    Parameters
    ----------
    x : (int, tuple)
        Item that needs to be converted to a tuple.
    """    
    return (x, x) if isinstance(x, int) else x
