# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_basics.ipynb.

# %% auto 0
__all__ = ['COORD_TYPES', 'ITER_TYPES', 'ITER_TYPES_TUPLE', 'ITER_TYPES_EXTRA', 'ALL_TYPES', 'BX_TYPE', 'Bx', 'BaseBx', 'MultiBx',
           'jbx', 'lbx', 'mbx', 'bbx', 'get_bx', 'stack_bxs', 'add_bxs', 'stack_bxs_inplace']

# %% ../nbs/01_basics.ipynb 2
import warnings
import inspect
from typing import Union
from numpy.typing import ArrayLike

import numpy as np
from fastcore.dispatch import explode_types
from fastcore.foundation import L, noop
from fastcore.basics import concat, store_attr, patch, GetAttr
from fastcore.xtras import is_listy

from pybx.ops import (
    mul,
    sub,
    intersection_box,
    make_single_iterable,
    voc_keys,
    update_keys,
)
from .excepts import *

COORD_TYPES = (np.float_, np.int_, int)
ITER_TYPES = (np.ndarray, list, L)
ITER_TYPES_TUPLE = (tuple,)
ITER_TYPES_EXTRA = (dict,)
ALL_TYPES = COORD_TYPES + ITER_TYPES

# %% ../nbs/01_basics.ipynb 5
class Bx:
    """Interface for all future Bx's"""

    def __init__(self, coords, label: list = None):
        label = label if label else []
        label = L(label) if not is_listy(label) else label
        coords = [coords] if len(coords) > 1 else coords  # make list of list
        # other props
        _coords = coords[0]  # internat representation as a list
        assert len(_coords) == 4, f"Expected 4 items in _coords, got {_coords}"
        x_min, y_min, x_max, y_max = _coords
        store_attr("x_min, y_min, x_max, y_max, _coords, coords, label")

    def __str__(self):
        return f"Bx(coords={self.coords}, label={self.label})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.coords)

    def get_coords(self):
        return self.coords

    @property
    def coords_as_numpy(self):
        return np.array(self.coords, dtype=int)

    def get_label(self):
        return self.label

    @property
    def bw(self):
        """Calculate width"""
        return sub(*self._coords[::2][::-1])

    @property
    def bh(self):
        """Calculate height"""
        return sub(*self._coords[1::2][::-1])

    @property
    def cx(self):
        """Calculate centroid-x"""
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self):
        """Calculate centroid-y"""
        return (self.y_min + self.y_max) / 2.0

    @property
    def area(self):
        """Calculates the absolute value of the area of the box."""
        return abs(mul(self.bw, self.bh))

    @property
    def values(self):
        """Returns the coordinates and label as a single list."""
        return L([[*self._coords, *self.label]])

    @property
    def valid(self):
        """Checks for validity of the box and returns a boolean.
        From `v0.1.3`, validity implies that the box has non-zero area.
        """
        v_area = bool(self.area)  # False if 0
        v_all = np.array([v_area])
        return True if v_all.all() else False

    @property
    def xywh(self):
        """Converts the `pascal_voc` bounding box to `coco` format."""
        return L([[self.x_min, self.y_min, self.bw, self.bh]])

    def yolo(self, w=1, h=1, normalize=False):
        """Converts the `pascal_voc` bounding box to `yolo` centroids format.
        :param normalize: Whether to normalize the bounding box with image width and height.
        :param w: Width of image. Not to be confused with `BaseBx` attribute `w`.
        :param h: Height of image. Not to be confused with `BaseBx` attribute `h`.
        """
        if normalize:
            assert (w > 1) and (
                (h > 1)
            ), f"{inspect.stack()[0][3]} of {__name__}: Expected width and height of image with normalize={normalize}."
        _yolo = np.array([[self.cx, self.cy, self.bw, self.bh]]) / np.tile([w, h], 2)
        return L(_yolo.round(4).tolist())

# %% ../nbs/01_basics.ipynb 22
class BaseBx(Bx):
    """BaseBx is the most primitive form of representing a bounding box.
    Coordinates and label of a bounding box can be wrapped as a BaseBx using:
    `bbx(coords, label)`.

    :param coords: can be of type `list` or `array` representing a single box.
        - `list` can be formatted with `label`: `[x_min, y_min, x_max, y_max, label]`
            or without `label`: `[x_min, y_min, x_max, y_max]`
        - `array` should be a 1-dimensional array of shape `(4,)`

    :param label: a `list` or `str` that has the class name or label for the object
    in the corresponding box.
    """

    def __init__(self, coords, label: list = None):
        self.index = 0  # Fixes #2, calls itself everytime
        assert isinstance(
            coords, (list, L, np.ndarray)
        ), f"{__name__}: Expected type list or np.ndarray for coords, got {type(coords)}"
        assert isinstance(coords[0], ALL_TYPES), (
            f"{__name__}: Expected float, int or single-nested list or np.ndarray at coords[0], "
            f"got {type(coords[0])} with {coords[0]}"
        )
        super().__init__(coords, label)

    def __str__(self):
        return f"BaseBx(coords={self.coords}, label={self.label})"

# %% ../nbs/01_basics.ipynb 32
@patch
def iou(self: BaseBx, other):
    """Caclulates the Intersection Over Union (IOU) of the box
    w.r.t. another `BaseBx`. Returns the IOU only if the box is
    considered `valid`.
    """
    if not isinstance(other, Bx):
        other = bbx(other)
    if self.valid:
        try:
            int_box = bbx(intersection_box(self.coords, other.coords))
        except NoIntersection:
            return 0.0
        int_area = int_box.area
        union_area = other.area + self.area - int_area
        return int_area / union_area
    return 0.0

# %% ../nbs/01_basics.ipynb 37
@patch
def __iter__(self: BaseBx):
    """Iterates through the boxes in `BaseBx` where self.valid is True."""
    return self


@patch
def __getitem__(self: BaseBx, idx):
    """Gets the item at index idx as a BaseBx."""
    if idx > 0:
        # Fixes #2
        raise IndexError(
            f"BaseBx has only a single coordinate at idx=0. Got idx={idx}."
        )
    return self


@patch
def __next__(self: BaseBx):
    """Iteration is allowed only for valid boxes"""
    try:
        b = self[self.index]
        if not b.valid:
            # 0 area boxes are not valid
            self.index += 1
            return self.__next__()
    except IndexError:
        self.index = 0  # reset index
        raise StopIteration
    self.index += 1
    return b

# %% ../nbs/01_basics.ipynb 42
class MultiBx:
    """`MultiBx` represents a collection of bounding boxes as ndarrays.
    Objects of type `MultiBx` can be indexed into, which returns a
    `BaseBx` exposing a suite of box-bound operations.
    Multiple coordinates and labels of bounding boxes can be wrapped
    as a `MultiBx` using:
        `mbx(coords, label)`.
    :param coords: can be nested coordinates of type `list` of `list`s/`json` records
        (`list`s of `dict`s)/`ndarray`s representing multiple boxes.
        If passing a list/json each index of the object should be of the following formats:
        - `list` can be formatted with `label`: `[x_min, y_min, x_max, y_max, label]`
            or without `label`: `[x_min, y_min, x_max, y_max]`
        - `dict` should be in `pascal_voc` format using the keys
            {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "label": 'none'}
        If passing an `ndarray`, it should be of shape `(N,4)`.

    :param label: a `list` of `str`s that has the class name or label for the object in the
    corresponding box.
    """

    def __init__(self, coords, label: list = None):
        index = 0
        coords = coords.tolist() if isinstance(coords, np.ndarray) else list(coords)
        if label is None:
            label = [None] * len(coords)
        assert len(coords) == len(
            label
        ), f"wrong shape for coords {len(coords)} and label {len(label)}"
        bxs = [BaseBx([list(c)], l) for c, l in zip(coords, label)]
        store_attr("coords, label, index, bxs")

    def __len__(self):
        """Gets the length of coordinates."""
        return len(self.bxs)

    def __getitem__(self, idx):
        """Gets the item at index idx as a BaseBx."""
        return self.bxs[idx]

    def __iter__(self):
        """Iterates through the boxes in `MultiBx` where self.valid is True."""
        return self

    def __next__(self):
        """Iteration is allowed only for valid boxes"""
        try:
            b = self[self.index]
            if not b.valid:
                # 0 area boxes are not valid
                self.index += 1
                return self.__next__()
        except IndexError:
            self.index = 0  # reset index
            raise StopIteration
        self.index += 1
        return b

    def __str__(self):
        return f"MultiBx(coords={self.coords}, label={self.label})"

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self):
        """Returns shape of the coordinates"""
        return self.coords.shape

# %% ../nbs/01_basics.ipynb 43
BX_TYPE = (Bx, MultiBx)

# %% ../nbs/01_basics.ipynb 60
class __JsonBx(MultiBx):
    """
    If five items per coordinate are passed, the last index is taken as the label.
    """

    def __init__(self, coords, label: list = None):
        super().__init__(coords, label)

    @classmethod
    def jsonbx(cls, coords, label=None, keys=None):
        """Classmethod for JsonBx.

        Also accepts keys as a list, otherwise uses `voc_keys`.
        """
        l = L()
        r = L()
        for i, c in enumerate(coords):
            assert isinstance(
                c, ITER_TYPES + ITER_TYPES_EXTRA
            ), f"{inspect.stack()[0][3]} of \
                {__name__}: Expected b of type dict, got {type(c)}"
            if keys is None:
                # Fixes issue #3.
                keys = update_keys(c, default_keys=voc_keys)
            c_ = [c[k] for k in keys]  # read in order
            l_ = c_[-1] if len(c_) > 4 else "" if label is None else label[i]
            l.append(l_)
            r.append(c_[:-1] if len(c_) > 4 else c_)
        return cls(r, label=l)

    def __str__(self):
        return f"__JsonBx(coords={self.coords}, label={self.label})"


def jbx(coords=None, labels=None, keys=None):
    """Alias of the JsonBx class to process `json` records into
    `MultiBx` or `BaseBx` objects exposing many validation methods

    Also accepts keys as a list, otherwise uses `voc_keys`.
    """
    return __JsonBx.jsonbx(coords, labels, keys)

# %% ../nbs/01_basics.ipynb 67
class __ListBx(MultiBx):
    """
    If five items per coordinate are passed, the last index is taken as the label.
    """

    def __init__(self, coords, label: list = None):
        super().__init__(coords, label)

    @classmethod
    def listbx(cls, coords, label=None):
        """Classmethod for __ListBx."""
        l = L()
        r = L()
        for i, c in enumerate(coords):
            assert isinstance(
                c, ITER_TYPES
            ), f"{inspect.stack()[0][3]} of \
                {__name__}: Expected b of type list, got {type(c)}"
            l_ = label[i] if label else c[-1] if len(c) > 4 else None
            l.append(l_)
            r.append(c[:-1] if len(c) > 4 else c)
        return cls(r, label=l)

    def __str__(self):
        return f"__ListBx(coords={self.coords}, label={self.label})"


def lbx(coords=None, labels=None):
    """Alias of the __ListBx class to process `list` into
    `MultiBx` or `BaseBx` objects exposing many validation methods
    """
    return __ListBx.listbx(coords, labels)

# %% ../nbs/01_basics.ipynb 72
@patch(cls_method=True)
def multibox(cls: MultiBx, coords, label: list = None, keys: list = None):
    """Classmethod for `MultiBx`. Same as mbx(coords, label).
    Calls classmethods of `JsonBx` and `ListBx` based on the type
    of coords passed.
    """
    t = explode_types(coords)
    # if explode_types returns a single class, it means they are not nested
    if t == np.ndarray:
        return cls(coords, label)
    if t == dict:
        b = jbx([coords], label, keys)
        return cls(b.coords, b.label)
    if t == list:
        b = lbx([coords], label)
        return cls(b.coords, b.label)
    # if list of list or dicts
    type_l0 = list(t.keys())[0]
    type_l1 = t[type_l0][0]
    # process the data
    if type_l1 == dict:
        b = jbx(coords, label, keys)
        return cls(b.coords, b.label)
    # process lists of lists or ndarray
    try:
        b = lbx(coords, label)
        return cls(b.coords, b.label)
    except:
        return cls(coords, label)


def mbx(coords=None, label=None, keys=None):
    """Alias of the `MultiBx` class."""
    return MultiBx.multibox(coords, label, keys)

# %% ../nbs/01_basics.ipynb 85
@patch(cls_method=True)
def basebx(cls: BaseBx, coords, label: list = None, keys: list = voc_keys):
    """Classmethod for `BaseBx`. Same as bbx(coords, label), made to work with
    other object types other than ndarray."""
    try:
        coords, label = make_single_iterable(coords, keys=keys)
    except ValueError:
        coords = make_single_iterable(coords)
    finally:
        return cls(coords, label)


def bbx(coords=None, labels=None, keys=voc_keys):
    """Alias of the `BaseBx` class."""
    return BaseBx.basebx(coords, labels, keys)

# %% ../nbs/01_basics.ipynb 98
def get_bx(coords, label=None):
    """
    Helper function to check and call the correct type of Bx instance.

    Checks for the type of data passed and calls the respective class
    to generate a Bx instance. Currently only supports ndarray, list, dict,
    tuple, nested list, nested tuple.

    Parameters
    ----------
    coords : ndarray, list, dict, tuple, nested list, nested tuple
        Coordinates of anchor boxes.
    label : list, optional
        Labels for anchor boxes in order, by default None

    Returns
    -------
    Bx
        An instance of MultiBx, ListBx, BaseBx or JsonBx

    Raises
    ------
    NotImplementedError
        If unknown type of coordinates are passed.
    """
    # process ndarray
    if isinstance(coords, np.ndarray):
        coords = np.atleast_2d(coords)
        return mbx(coords, label)
    # process list
    if isinstance(coords, (list, L)):
        if isinstance(coords[0], COORD_TYPES):
            """If first item is a position"""
            return bbx(coords, label)
        elif isinstance(coords[0], ITER_TYPES + ITER_TYPES_EXTRA):
            """If fist item is an iterable"""
            return mbx(coords, label)
        elif isinstance(coords[0], ITER_TYPES_TUPLE):
            """If first item is a tuple"""
            return mbx([list(c) for c in coords], label)
    # process dict
    if isinstance(coords, dict):
        return bbx([coords], label)
    # process tuple
    if isinstance(coords, tuple):
        return bbx(list(coords), label)
    # process BX_TYPE
    if isinstance(coords, BX_TYPE):
        return coords
    else:
        raise NotImplementedError(
            f"{inspect.stack()[0][3]} of {__name__}: Got coords={coords} of type {type(coords)}."
        )

# %% ../nbs/01_basics.ipynb 107
@patch
def __add__(self: BaseBx, other):
    """Pseudo-add method that stacks the provided boxes and labels. Stacking two
    boxes imply that the resulting box is a `MultiBx`: `BaseBx` + `BaseBx`
    = `MultiBx`. This violates the idea of `BaseBx` since the result
    holds more than 1 coordinate/label for the box.
    From `v.2.0`, a `UserWarning` is issued if called.
    Recommended use is either: `BaseBx` + `BaseBx` = `MultiBx` or
    `basics.stack_bxs()`.
    """
    if not isinstance(other, BX_TYPE):
        raise TypeError(
            f"{inspect.stack()[0][3]} of {__name__}: Expected a subclass of {BX_TYPE}"
        )
    else:
        warnings.warn(
            BxViolation(
                f"Change of object type imminent if trying to add "
                f"{type(self)}+{type(other)}. Use {type(other)}+{type(self)} "
                f"instead or basics.stack_bxs()."
            )
        )
    coords = self.coords + other.coords
    label = self.label + other.label
    return mbx(coords, label)


@patch
def __add__(self: MultiBx, other):
    """Pseudo-add method that stacks the provided boxes and labels. Stacking two
    boxes imply that the resulting box is a `MultiBx`: `MultiBx` + `MultiBx`
    = `MultiBx`. Same as `basics.stack_bxs()`.
    """
    if not isinstance(other, BX_TYPE):
        raise TypeError(
            f"{inspect.stack()[0][3]} of {__name__}: Expected type {BX_TYPE}, "
            f"got self={type(self)}, other={type(other)}"
        )
    coords = self.coords + other.coords
    label = self.label + other.label
    return mbx(coords, label)

# %% ../nbs/01_basics.ipynb 108
def stack_bxs(b1, b2):
    """
    Method to stack two Bx-types together. Similar to `__add__` of BxTypes
    but avoids UserWarning.
    :param b1:
    :param b2:
    :return:
    _summary_

    Parameters
    ----------
    b1 : Bx, MultiBx
        Anchor box coordinates Bx
    b2 : Bx, MultiBx
        Anchor box coordinates Bx

    Returns
    -------
    MultiBx
        Stacked anchor box coordinates of MultiBx type.

    Raises
    ------
    TypeError
        If unknown type of coordinates are passed.
    """

    if not isinstance(b1, BX_TYPE):
        raise TypeError(
            f"{inspect.stack()[0][3]} of {__name__}: Expected type {BX_TYPE}, got b1={type(b1)}"
        )
    if not isinstance(b2, BX_TYPE):
        raise TypeError(
            f"{inspect.stack()[0][3]} of {__name__}: Expected type {BX_TYPE}, got b2={type(b2)}"
        )
    if isinstance(b1, BaseBx):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return b1 + b2
    return b1 + b2


def add_bxs(b1, b2):
    """Alias of stack_bxs()."""
    return stack_bxs(b1, b2)

# %% ../nbs/01_basics.ipynb 118
def stack_bxs_inplace(b, *args):
    """Stack the passed boxes on top of the first item."""
    for b_ in args:
        b = stack_bxs(b, b_)
    return b

# %% ../nbs/01_basics.ipynb 127
@patch
def get_offset(
    self: BaseBx,
    other: BaseBx,
    normalize=True,
    log_func=np.log,
    sigma=(0.1, 0.2),
    self_is_anchor=False,
):
    """
    Caclulates the offset of the box I with another box O.
    The most basic calculation of offset involves a) taking the distance between the centers: `I_cx - O_cx`, `I_cy - O_cy`.
    b) taking the ratio of the two boxes: `I_w/Ow, I_h/O_h`.

    If `normalize=True`, the center distances and ratios are normalized as per https://arxiv.org/pdf/1512.02325.pdf
    `(I_cx - O_cx)/O_w`, `(I_cy - O_cy)/O_h`, `log(I_w/Ow), log(I_h/O_h)`
    These are further scaled with an appoximation of standard deviation for the distances and ratios
    `((I_cx - O_cx)/O_w)/sigma_c`, `((I_cy - O_cy)/O_h)/sigma_c`, `log(I_w/Ow)/sigma_r, log(I_h/O_h)/sigma_r`

    Args:
        other (BaseBx): Any supported type of bounding box format, even takes a list of coordinates. Typically the anchor box.
        normalize (bool, optional): Whether to normalize the offsets. Defaults to True.
        log_func (func, optional): Function for normalizing the ratio of widths and heights. Defaults to np.log.
        sigma (tuple, optional): Estimated of standard deviation for the distances and ratios. Defaults to (0.1, 0.2).
        self_is_anchor (bool, optional): Typically `other` is assumed to be the anchor box, this flag tells that this assumption is False. Defaults to False.

    Returns:
        list: Offsets of the two bounding boxes
    """
    if isinstance(other, MultiBx):
        warnings.warn(BxViolation(f"Other should be BaseBx, got MultiBx"))
        assert len(other) == 1, f"{other} cannot be converted to single bounding box."
        other = other[0]
    elif not isinstance(other, Bx):
        other = bbx(other)

    if self_is_anchor:
        # if self_is_anchor, ie anchor.get_offset(ground_truth) is called
        anchor = self
        gt = other
    else:
        # if not self_is_anchor, ie ground_truth.get_offset(anchor) is called (default behaviour)
        gt = self
        anchor = other
    # get anchor box w and h
    anchor_bw_norm = anchor.bw
    anchor_bh_norm = anchor.bh
    sigma_c, sigma_r = sigma
    # if not normalize, reset params
    if not normalize:
        log_func = noop
        sigma_c, sigma_r, anchor_bw_norm, anchor_bh_norm = [1.0] * 4
    # center distances
    # norm with anchor box w and h
    cx_offset, cy_offset = (gt.cx - anchor.cx) / anchor_bw_norm, (
        gt.cy - anchor.cy
    ) / anchor_bh_norm
    # scale of boxes
    w_offset = log_func(gt.bw / anchor_bw_norm)
    h_offset = log_func(gt.bh / anchor_bh_norm)

    offset = np.asarray([cx_offset, cy_offset, w_offset, h_offset])
    # norm with sigmaxy and sigmawh
    # print(sigma_c, sigma_r)
    offset /= np.repeat([sigma_c, sigma_r], 2)
    # not np.tile as norm is cx/sigma_c, cy/sigma_c, w/sigma_r, h/sigma_r
    return L(offset.round(4).tolist())
