# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_anchor.ipynb.

# %% auto 0
__all__ = ['bx', 'bxs', 'get_gt_thresh_iou', 'get_gt_max_iou', 'get_gt_offsets']

# %% ../nbs/00_anchor.ipynb 2
import inspect

import math
import numpy as np
from fastcore.foundation import L, mask2idxs
from fastcore.utils import gt
from numpy.typing import ArrayLike
from typing import Union
import json
from collections import defaultdict
import warnings

from .ops import named_idx
from .basics import get_bx, stack_bxs_inplace, BX_TYPE, BaseBx, Bx, bbx, get_bx
from .utils import get_edges, validate_boxes, as_tuple, reassign_label
from .excepts import NoGroundTruthBxs

# %% ../nbs/00_anchor.ipynb 4
def bx(
    image_sz: (int, tuple),
    feature_sz: (int, tuple),
    asp_ratio: float = None,
    clip: bool = True,
    named: bool = True,
    anchor_sfx: str = "a",
    min_visibility: float = 0.25,
) -> ArrayLike:
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
    ArrayLike
        anchor box coordinates in `pascal_voc` format
        if named=True, a list of anchor box labels are also returned.
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
    b = validate_boxes(
        coords, image_sz, feature_sz, clip=clip, min_visibility=min_visibility
    )
    if named:
        anchor_sfx = f"{anchor_sfx}_{feature_sz[0]}x{feature_sz[1]}_{asp_ratio:.1f}_"
        labels = named_idx(len(b), anchor_sfx)
    # init multibx
    b = get_bx(b, labels)
    return (b.coords, b.label) if named else b.coords

# %% ../nbs/00_anchor.ipynb 8
def bxs(
    image_sz: (int, tuple),
    feature_szs: list = None,
    asp_ratios: list = None,
    named: bool = True,
    **kwargs,
) -> ArrayLike:
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
    ArrayLike
        anchor box coordinates in pascal_voc format
        if named=True, a list of anchor box labels are also returned.
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

# %% ../nbs/00_anchor.ipynb 45
def get_gt_thresh_iou(
    true_annots,
    anchor_boxes,
    anchor_labels=None,
    iou_thresh=0.3,
    return_ious=False,
    return_masks=False,
    update_labels=True,
):
    """Calculate positive ground truth and extra positive ground truth bounding boxes based on iou threhsold.

    Can result in uneven number of positive ground truth boxes per class.

    Args:
        true_annots (Any): True annotations, typically in `pascal_voc` format
        anchor_boxes (Any): Candidate anchor boxes, typically calculated with `pybx.bxs`
        anchor_labels (List, optional): Anchor box labels, will be overwritten with true labels if `update_labels=True`. Defaults to None.
        iou_thresh (float, optional): IOU threshold to filter out negative ground truth anchor boxes. Defaults to 0.3.
        return_ious (bool, optional): Return IOU values for selected positive ground truth anchor boxes. Defaults to False.
        return_masks (bool, optional): Return boolean masks for all anchor boxes indicating if a box is positive (`True`) or negative (`False`) ground truth box. Defaults to False.
        update_labels (bool, optional): Overwrite with true annotations. Defaults to True.

    Returns:
        dict: positive ground truth anchor boxes per class
        dict: IOU of positive ground truth anchor boxes per class
        dict: boolean list indicating positive ground truth anchor boxes per class
    """
    gt_anchors_per_class = defaultdict(lambda: L())
    iou_per_class = defaultdict(lambda: L())
    mask_per_class = defaultdict(lambda: L())
    true_annots_as_bx = (
        get_bx(true_annots) if not isinstance(true_annots, BX_TYPE) else true_annots
    )
    anchor_labels = (
        [f"bx_{i}" for i in range(len(anchor_boxes))]
        if anchor_labels is None
        else anchor_labels
    )
    coords_as_bx = (
        get_bx(coords=anchor_boxes, label=anchor_labels)
        if not isinstance(anchor_boxes, BX_TYPE)
        else anchor_boxes
    )
    n_boxes = len(coords_as_bx)

    for annots in true_annots_as_bx:
        label = annots.label[0]  # is a list of len 1
        ious = L([round(annots.iou(coords_as_bx[i]), 4) for i in range(n_boxes)])
        # ious_filter = ious.argwhere(gt(iou_thresh))
        mask = ious.map(lambda x: x >= iou_thresh)
        ious_filter = mask2idxs(mask=mask)

        if mask.sum() < 1:
            warnings.warn(
                NoGroundTruthBxs(
                    f"No good ground truth anchors found for label={label}, try lowering threshold (iou_thresh={iou_thresh} or increasing candidates."
                )
            )
            gt_anchors_per_class[label] = None

        if return_ious:
            # report filtered box IOUs
            iou_per_class[label].extend(ious[ious_filter])
        if return_masks:
            mask_per_class[label].extend(mask)
        # report selected boxes
        # print([coords_as_bx[i] for i in ious_filter])
        if mask.sum() > 0:
            gt_anchors_per_class[label] = stack_bxs_inplace(
                *[
                    reassign_label(coords_as_bx[i], label=[label])
                    if update_labels
                    else coords_as_bx[i]
                    for i in ious_filter
                ]
            )

    return dict(gt_anchors_per_class), dict(iou_per_class), dict(mask_per_class)

# %% ../nbs/00_anchor.ipynb 59
def get_gt_max_iou(
    true_annots,
    anchor_boxes,
    anchor_labels=None,
    return_ious=False,
    return_masks=False,
    positive_boxes=1,
    update_labels=True,
):
    """Calculate positive ground truth and extra positive ground truth bounding boxes based on maximum IOU condition.

    Will always provide a box, therfore constant number `positive_boxes` of positive ground truth boxes per class.

    Args:
        true_annots (Any): True annotations, typically in `pascal_voc` format
        anchor_boxes (Any): Candidate anchor boxes, typically calculated with `pybx.bxs`
        anchor_labels (List, optional): Anchor box labels, will be overwritten with true labels if `update_labels=True`. Defaults to None.
        return_ious (bool, optional): Return IOU values for selected positive ground truth anchor boxes. Defaults to False.
        return_masks (bool, optional): Return boolean masks for all anchor boxes indicating if a box is positive (`True`) or negative (`False`) ground truth box. Defaults to False.
        update_labels (bool, optional): Overwrite with true annotations. Defaults to True.
        positive_boxes (int, optional): Number of extra/positive ground truth boxes to return. Defaults to 1.

    Returns:
        dict: positive ground truth anchor boxes per class
        dict: IOU of positive ground truth anchor boxes per class
        dict: boolean list indicating positive ground truth anchor boxes per class
    """
    gt_anchors_per_class = defaultdict(lambda: L())
    iou_per_class = defaultdict(lambda: L())
    mask_per_class = defaultdict(lambda: L())
    true_annots_as_bx = (
        get_bx(true_annots) if not isinstance(true_annots, BX_TYPE) else true_annots
    )
    anchor_labels = (
        [f"bx_{i}" for i in range(len(anchor_boxes))]
        if anchor_labels is None
        else anchor_labels
    )
    coords_as_bx = (
        get_bx(coords=anchor_boxes, label=anchor_labels)
        if not isinstance(anchor_boxes, BX_TYPE)
        else anchor_boxes
    )
    n_boxes = len(coords_as_bx)

    for annots in true_annots_as_bx:
        label = annots.label[0]  # is a list of len 1
        ious = L([round(annots.iou(coords_as_bx[i]), 4) for i in range(n_boxes)])
        ious_sorted = ious.sorted(reverse=True)
        max_iou = ious_sorted[:positive_boxes]
        ious_filter = [ious.index(m) for m in max_iou]
        mask = L([True if idx in ious_filter else False for idx in range(n_boxes)])

        if mask.sum() < 1:
            warnings.warn(
                NoGroundTruthBxs(
                    f"No good ground truth anchors found for label={label}, try increasing candidates."
                )
            )
            gt_anchors_per_class[label] = None

        # print(max_iou, ious_filter)
        if return_ious:
            # report filtered box IOUs
            iou_per_class[label].extend(ious[ious_filter])
        if return_masks:
            mask_per_class[label].extend(mask)
        # report selected boxes
        # print([coords_as_bx[i] for i in ious_filter])
        if mask.sum() > 0:
            gt_anchors_per_class[label] = stack_bxs_inplace(
                *[
                    reassign_label(coords_as_bx[i], label=[label])
                    if update_labels
                    else coords_as_bx[i]
                    for i in ious_filter
                ]
            )

    return dict(gt_anchors_per_class), dict(iou_per_class), dict(mask_per_class)

# %% ../nbs/00_anchor.ipynb 84
def get_gt_offsets(
    true_annots: BaseBx,
    anchor_boxes,
    anchor_labels=None,  # do we need to pass this
    masks=None,
    sigma=(0.1, 0.2),
    normalize=True,
    log_func=np.log,
    update_labels=False,
):
    if not isinstance(true_annots, Bx):
        true_annots = bbx(true_annots)

    Na = len(
        anchor_boxes
    )  # no of anchor boxes (includes positive and negative anchor boxes)
    masks = (
        masks if masks is not None else L([True] * Na)
    )  # if no masks provided, repeat for all anchors
    offsets = np.zeros((Na, 4))
    labels = L(["background"] * Na)  # if update_labels else anchor_labels
    true_label = true_annots.label[0] if len(true_annots.label) != 0 else "unknown"
    for idx, (box, mask) in enumerate(zip(anchor_boxes, masks)):
        if mask:
            offsets[idx, :] = true_annots.get_offset(
                box, normalize=normalize, sigma=sigma, log_func=log_func
            )
            # labels with mask=True will be the candidates for actual ground truth class.
            if update_labels:  # and (labels is not None):
                labels[idx] = true_label
    return offsets, labels
