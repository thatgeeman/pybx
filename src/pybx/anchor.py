import inspect

import math
import numpy as np
from fastcore.foundation import L

from .ops import __ops__, get_op, named_idx
from .basics import get_bx


def get_edges(image_sz: tuple, feature_sz: tuple, op='noop'):
    """Generate offsetted top `(x_min, y_min)` or bottom edges `(x_max, y_max)`
    coordinates of a given feature size based on op.
    if `op` is `noop`, gets the top edges.
    if `op` is `add`, gets the bottom edges.
    :param op: operation for calculating edges, either 'add' 'sub' 'noop'
    :param image_sz: tuple of `(W, H, C)` of an image
    :param feature_sz: tuple of `(W, H)` of a channel
    :return: offsetted edges of each feature
    """
    assert image_sz[-1] < image_sz[
        0], f'{inspect.stack()[0][3]} of {__name__}: Expected {image_sz[-1]} < {image_sz[0]}={image_sz[1]}'
    assert len(image_sz) == 3, f'{inspect.stack()[0][3]} of {__name__}: Expected image_sz of len 3, got {len(image_sz)}'
    assert op in __ops__, f'{inspect.stack()[0][3]} of {__name__}: Operator not in allowed operations: {__ops__}'
    w, h, _ = image_sz
    nx, ny = feature_sz
    diag_edge_ofs = w / nx, h / ny
    op_ = get_op(op)
    x_ = op_(np.linspace(0, w, nx + 1), diag_edge_ofs[0])
    y_ = op_(np.linspace(0, h, ny + 1), diag_edge_ofs[1])
    mesh = np.meshgrid(x_, y_)
    edges = np.stack([m.flatten() for m in mesh], axis=-1)
    return edges


def validate_boxes(coords, image_sz, feature_sz, clip=True, min_vis=0.25):
    """Validate calculated anchor box coords.
    :param coords: anchor box coordinates
    :param image_sz: tuple of (width, height) of an image
    :param feature_sz: tuple of (width, height) of a channel
    :param clip: whether to apply np.clip
    :param min_vis: minimum visibility dictates the condition for a box to be considered valid. The value corresponds to the
    ratio of expected area to the calculated area after clipping to image dimensions.
    :return: anchor box coordinates in [pascal_voc] format
    """
    _max = max(image_sz[0], image_sz[1])
    # clip the boxes to image dimensions
    b = get_bx(coords.clip(0, _max)) if clip else get_bx(coords)
    # check if the area of the bounding box is fitting the minimum area criterion
    min_area = (image_sz[0] / feature_sz[0]) * (image_sz[1] / feature_sz[1]) * min_vis
    b = [b_.coords for b_ in b if b_.area() > min_area]
    return b


def bx(image_sz: tuple, feature_sz: tuple, asp_ratio: float = None, clip=True, named=True, anchor_sfx: str = 'a', min_vis=0.25):
    """Calculate anchor box coords given an image size and feature size for a single aspect ratio.
    :param image_sz: tuple of (width, height) of an image
    :param feature_sz: tuple of (width, height) of a channel
    :param asp_ratio: aspect ratio (width:height)
    :param clip: whether to apply np.clip
    :param named: whether to return (coords, labels)
    :param anchor_sfx: suffix for anchor label: anchor_sfx_asp_ratio_feature_sz
    :param min_vis: minimum visibility dictates the condition for a box to be considered valid. The value corresponds to the
    ratio of expected area to the calculated area after clipping to image dimensions.
    :return: anchor box coordinates in [pascal_voc] format
    """
    assert image_sz[-1] < image_sz[0], f'{inspect.stack()[0][3]} of {__name__}: \
    Expected {image_sz[-1]} < {image_sz[0]}={image_sz[1]}'
    labels = None
    asp_ratio = 1. if asp_ratio is None else asp_ratio
    # n_boxes = __mul__(*feature_sz)
    top_edges = get_edges(image_sz, feature_sz, op='noop')
    bot_edge = get_edges(image_sz, feature_sz, op='add')
    coords = np.hstack([top_edges, bot_edge])  # raw coords
    coords_wh = (coords[:, 2:] - coords[:, :2])  # w -> xmax-xmin, h -> ymax-ymin
    coords_center = (coords[:, 2:] - coords_wh / 2)  # xmax-w/2, ymax-h/2
    # scale the dimension of width and height with asp ratios
    _w = coords_wh[:, 0] * math.sqrt(asp_ratio)
    _h = coords_wh[:, 1] / math.sqrt(asp_ratio)
    coords_asp_wh = np.stack([_w, _h], -1)
    xy_min = coords_center - coords_asp_wh / 2
    xy_max = coords_center + coords_asp_wh / 2
    coords = np.hstack([xy_min, xy_max])
    # check for valid boxes
    b = validate_boxes(coords, image_sz, feature_sz, clip=clip, min_vis=min_vis)
    if named:
        anchor_sfx = f'{anchor_sfx}_{feature_sz[0]}x{feature_sz[1]}_{asp_ratio:.1f}_'
        labels = named_idx(len(b), anchor_sfx)
    # init multibx
    b = get_bx(b, labels)
    return (b.coords, b.label) if named else b.coords


def bxs(image_sz, feature_szs: list = None, asp_ratios: list = None, named: bool = True, **kwargs):
    """Calculate anchor box coords given an image size and multiple feature sizes for mutiple aspect ratios.
    :param image_sz: tuple of (width, height) of an image
    :param feature_szs: list of feature sizes for anchor boxes, each feature size being a tuple of (width, height) of a channel
    :param asp_ratios: list of aspect rations for anchor boxes, each aspect ratio being a float calculated by (width:height)
    :param named: whether to return (coords, labels)
    :return: anchor box coordinates in [pascal_voc] format
    """
    assert image_sz[-1] < image_sz[
        0], f'{inspect.stack()[0][3]} of {__name__}: Expected {image_sz[-1]} < {image_sz[0]}={image_sz[1]}'
    asp_ratios = [1 / 2., 1., 2.] if asp_ratios is None else asp_ratios
    feature_szs = [(8, 8), (2, 2)] if feature_szs is None else feature_szs
    # always named=True for bx() call. named=True in fn signature of bxs() is in its scope.
    coords_ = [bx(image_sz, f, ar, named=True, **kwargs) for f in feature_szs for ar in asp_ratios]
    coords_, labels_ = L(zip(*coords_))
    return (np.vstack(coords_), L([l_ for lab_ in labels_ for l_ in lab_])) if named else np.vstack(coords_)
