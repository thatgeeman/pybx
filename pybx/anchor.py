# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_anchor.ipynb.

# %% auto 0
__all__ = ['bx', 'bxs']

# %% ../nbs/00_anchor.ipynb 2
import inspect

import math
import numpy as np
from fastcore.foundation import L

from .ops import named_idx
from .basics import get_bx
from .utils import get_edges, validate_boxes, as_tuple

# %% ../nbs/00_anchor.ipynb 4
def bx(
    image_sz: (int,tuple),
    feature_sz: (int,tuple),
    asp_ratio: float = None,
    clip: bool = True,
    named: bool = True,
    anchor_sfx: str = "a",
    min_visibility: float = 0.25,
):
    """Calculate anchor box coords given an image size and feature size 
    for a single aspect ratio.

    Parameters
    ----------
    image_sz : (int,tuple)
        image size (width, height)
    feature_sz : (int,tuple)
        feature map size (width, height) 
    asp_ratio : float, optional
        aspect ratio (width:height), by default None
    clip : bool, optional
        whether to apply np.clip, by default True
    named : bool, optional
        whether to return (coords, labels), by default True
    anchor_sfx : str, optional
        suffix anchor label with anchor_sfx, by default "a"
    min_visibility : float, optional
        minimum visibility dictates the condition for a box to be considered 
        valid. The value corresponds to the ratio of expected area of an anchor box
        to the calculated area after clipping to image dimensions., by default 0.25

    Returns
    -------
    ndarray
        anchor box coordinates in `pascal_voc` format
    """
    labels = None
    image_sz = as_tuple(image_sz)
    feature_sz = as_tuple(feature_sz)
    asp_ratio = 1.0 if asp_ratio is None else asp_ratio
    # n_boxes = __mul__(*feature_sz)
    top_edges = get_edges(image_sz, feature_sz, op="noop")
    bot_edge = get_edges(image_sz, feature_sz, op="add")
    coords = np.hstack([top_edges, bot_edge])  # raw coords
    coords_wh = coords[:, 2:] - coords[:, :2]  # w -> xmax-xmin, h -> ymax-ymin
    coords_center = coords[:, 2:] - coords_wh / 2  # xmax-w/2, ymax-h/2
    # scale the dimension of width and height with asp ratios
    _w = coords_wh[:, 0] * math.sqrt(asp_ratio)
    _h = coords_wh[:, 1] / math.sqrt(asp_ratio)
    coords_asp_wh = np.stack([_w, _h], -1)
    xy_min = coords_center - coords_asp_wh / 2
    xy_max = coords_center + coords_asp_wh / 2
    coords = np.hstack([xy_min, xy_max])
    # check for valid boxes 
    b = validate_boxes(coords, image_sz, feature_sz, clip=clip, min_visibility=min_visibility)
    if named:
        anchor_sfx = f"{anchor_sfx}_{feature_sz[0]}x{feature_sz[1]}_{asp_ratio:.1f}_"
        labels = named_idx(len(b), anchor_sfx)
    # init multibx
    b = get_bx(b, labels)
    return (b.coords, b.label) if named else b.coords

# %% ../nbs/00_anchor.ipynb 7
def bxs(
    image_sz: (int,tuple),
    feature_szs: list = None,
    asp_ratios: list = None,
    named: bool = True,
    **kwargs,
):
    """Calculate anchor box coords given an image size and multiple 
    feature sizes for mutiple aspect ratios.

    Parameters
    ----------
    image_sz : (int,tuple)
        image size (width, height)
    feature_szs : list, optional
        list of feature map sizes, each feature map size being an int or tuple, by default [(8, 8), (2, 2)]
    asp_ratios : list, optional
        list of aspect ratios for anchor boxes, each aspect ratio being a float calculated by (width:height), by default [1 / 2.0, 1.0, 2.0]
    named : bool, optional
        whether to return (coords, labels), by default True

    Returns
    -------
    ndarray
        anchor box coordinates in pascal_voc format
    """ 
    image_sz = as_tuple(image_sz)
    feature_szs = [8, 2] if feature_szs is None else feature_szs
    feature_szs = [as_tuple(fsz) for fsz in feature_szs]
    asp_ratios = [1 / 2.0, 1.0, 2.0] if asp_ratios is None else asp_ratios
    # always named=True for bx() call. named=True in fn signature of bxs() is in its scope.
    coords_ = [
        bx(image_sz, f, ar, named=True, **kwargs)
        for f in feature_szs
        for ar in asp_ratios
    ]
    coords_, labels_ = L(zip(*coords_))
    coords_ = np.vstack(coords_)
    labels_ = L([l_ for lab_ in labels_ for l_ in lab_])
    return (coords_, labels_) if named else np.vstack(coords_)