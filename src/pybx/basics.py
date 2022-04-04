import warnings
import inspect

import numpy as np
from fastcore.basics import concat, store_attr
from fastcore.xtras import is_listy

from .ops import mul, sub, intersection_box, make_array, voc_keys, update_keys
from .excepts import *

__all__ = ['bbx', 'mbx', 'jbx', 'lbx', 'get_bx',
           'stack_bxs', 'add_bxs',
           'BaseBx', 'MultiBx', 'JsonBx', 'ListBx']


class BaseBx:
    """BaseBx is the most primitive form of representing a bounding box.
    Coordinates and label of a bounding box can be wrapped as a BaseBx using:
    `bbx(coords, label)`.

    :param coords: can be of type `list`/`dict`/`json`/`array` representing a single box.
        - `list` can be formatted with `label`: `[x_min, y_min, x_max, y_max, label]`
            or without `label`: `[x_min, y_min, x_max, y_max]`
        - `dict` should be in `pascal_voc` format using the keys
            {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "label": 'none'}
        - `json` records  should be a single-object `list` in `pascal_voc` format
           [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "label": 'none'}]
        - `array` should be a 1-dimensional array of shape `(1,4)` or `(4,)`

    :param label: a `list` or `str` that has the class name or label for the object
    in the corresponding box.
    """

    def __init__(self, coords, label: list = None):
        label = [label] if not is_listy(label) else label
        assert isinstance(coords, (list, np.ndarray)), \
            f'{inspect.stack()[0][3]} of {__name__}: Expected type {list}/{np.ndarray} for coords, got {type(coords)}'
        assert isinstance(coords[0], (np.floating, np.int_, np.ndarray, list)), \
            f'{inspect.stack()[0][3]} of {__name__}: Expected {float, int} or single-nested {list, np.ndarray} at coords[0], ' \
            f'got {type(coords[0])} with {coords[0]}'
        w = sub(*coords[::2][::-1])
        h = sub(*coords[1::2][::-1])
        x_min, y_min, x_max, y_max = coords
        cx = x_min + w / 2
        cy = y_min + h / 2
        store_attr('coords, label, x_min, y_min, x_max, y_max, w, h, cx, cy')

    def values(self):
        """Returns the coordinates and label as a single list."""
        return [*self.coords, *self.label]

    def area(self):
        """Calculates the absolute value of the area of the box."""
        return abs(mul(self.w, self.h))

    def iou(self, other):
        """Caclulates the Intersection Over Union (IOU) of the box
        w.r.t. another `BaseBx`. Returns the IOU only if the box is
        considered `valid`.
        """
        if not isinstance(other, BaseBx):
            other = bbx(other)
        if self.valid():
            try:
                int_box = bbx(intersection_box(self.coords, other.coords))
            except NoIntersection:
                return 0.0
            int_area = int_box.area()
            union_area = other.area() + self.area() - int_area
            return int_area / union_area
        return 0.0

    def valid(self):
        """Checks for validity of the box and returns a boolean.
        From `v0.1.3`, validity implies that the box has non-zero area.
        """
        v_area = bool(self.area())  # False if 0
        v_all = np.array([v_area])
        return True if v_all.all() else False

    def xywh(self):
        """Converts the `pascal_voc` bounding box to `coco` format."""
        return np.asarray(concat([self.x_min, self.y_min, self.w, self.h]))

    def yolo(self, normalize=False, w=None, h=None):
        """Converts the `pascal_voc` bounding box to `yolo` centroids format.
        :param normalize: Whether to normalize the bounding box with image width and height.
        :param w: Width of image. Not to be confused with `BaseBx` attribute `w`.
        :param h: Height of image. Not to be confused with `BaseBx` attribute `h`.
        """
        if normalize:
            assert w is not None, f'{inspect.stack()[0][3]} of {__name__}: Expected width and height of image with normalize={normalize}.'
            assert h is not None, f'{inspect.stack()[0][3]} of {__name__}: Expected width and height of image with normalize={normalize}.'
            return np.asarray(concat([self.cx / w, self.cy / h, self.w / w, self.h / h]))
        return np.asarray(concat([self.cx, self.cy, self.w, self.h]))

    def __len__(self):
        return self.coords.shape[0] + 1

    def __iter__(self):
        """Fake iterator fix for issue #2"""
        self.index = 0
        return self

    def __next__(self):
        if self.index == 0:
            self.index += 1
            return self
        raise StopIteration

    def __add__(self, other):
        """Pseudo-add method that stacks the provided boxes and labels. Stacking two
        boxes imply that the resulting box is a `MultiBx`: `BaseBx` + `BaseBx`
        = `MultiBx`. This violates the idea of `BaseBx` since the result
        holds more than 1 coordinate/label for the box.
        From `v.2.0`, a `UserWarning` is issued if called.
        Recommended use is either: `BaseBx` + `BaseBx` = `MultiBx` or
        `basics.stack_bxs()`.
        """
        if not isinstance(other, (BaseBx, MultiBx, JsonBx, ListBx)):
            raise TypeError(f'{inspect.stack()[0][3]} of {__name__}: Expected type MultiBx/JsonBx/ListBx')
        if isinstance(other, (BaseBx, MultiBx, JsonBx, ListBx)):
            warnings.warn(BxViolation(f'Change of object type imminent if trying to add '
                                      f'{type(self)}+{type(other)}. Use {type(other)}+{type(self)} '
                                      f'instead or basics.stack_bxs().'))
        coords = np.vstack([self.coords, other.coords])
        label = self.label + other.label
        return mbx(coords, label)

    @classmethod
    def basebx(cls, coords, label: list = None):
        """Classmethod for `BaseBx`. Same as bbx(coords, label)"""
        if not isinstance(coords, np.ndarray):
            """Process list/dict assuming a single coordinate is present"""
            try:
                coords, label = make_array(coords)
            except ValueError:
                coords = make_array(coords)
            except NotImplementedError:
                """Attempt to Process `list` of `list`s/`dict`s with len=1"""
                if len(coords) > 1:
                    raise BxViolation(f'{inspect.stack()[0][3]} of {__name__}: Expected single element in coords, got {coords}')
                try:
                    b = jbx(coords, label)
                except AssertionError:
                    b = lbx(coords, label)
                return cls(b.coords[0], b.label)
        return cls(coords, label)


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
        if isinstance(coords, np.ndarray):
            if label is None:
                label = [''] * coords.shape[0]
            assert coords.shape[0] == len(label), f'wrong shape for coords {coords.shape[0]} and label {len(label)}'
        store_attr('coords, label')

    def __getitem__(self, idx):
        """Gets the item at index idx as a BaseBx."""
        return BaseBx(self.coords[idx], label=self.label[idx])

    def __len__(self):
        return len(self.coords) + len(self.label)

    def __iter__(self):
        """Iterates through the boxes in `MultiBx` where self.valid() is True."""
        self.index = 0
        return self

    def __next__(self):
        try:
            b = self[self.index]
            if not b.valid():
                # 0 area boxes are not valid
                self.index += 1
                return self.__next__()
        except IndexError:
            raise StopIteration
        self.index += 1
        return b

    def __add__(self, other):
        """Pseudo-add method that stacks the provided boxes and labels. Stacking two
        boxes imply that the resulting box is a `MultiBx`: `MultiBx` + `MultiBx`
        = `MultiBx`. Same as `basics.stack_bxs()`.
        """
        if not isinstance(other, (BaseBx, MultiBx, JsonBx, ListBx)):
            raise TypeError(f'{inspect.stack()[0][3]} of {__name__}: Expected type BaseBx/MultiBx/JsonBx/ListBx, '
                            f'got self={type(self)}, other={type(other)}')
        coords = np.vstack([self.coords, other.coords])
        label = self.label + other.label
        return mbx(coords, label)

    @property
    def shape(self):
        """Returns shape of the coordinates"""
        return self.coords.shape

    @classmethod
    def multibox(cls, coords, label: list = None):
        """Classmethod for `MultiBx`. Same as mbx(coords, label).
        Calls classmethods of `JsonBx` and `ListBx` based on the type
        of coords passed.
        """
        if isinstance(coords, list):
            try:
                b = JsonBx.jsonbx(coords, label)
            except AssertionError:
                # if dict assertion fails
                b = ListBx.listbx(coords, label)
            return cls(b.coords, b.label)
        return cls(coords, label)


class ListBx:
    """`ListBx` represents a collection of bounding boxes as a `list` of `list`s. Internally
     called by `MultiBx` to process coordinates in the format `lists` of `list`s.

    :param coords: can be nested coordinates of type `list` of `list`s representing
    multiple boxes. If passing a `list`, each index of the object should be of the following
    format:`list` can be formatted with `label`: `[x_min, y_min, x_max, y_max, label]`
    or without `label`: `[x_min, y_min, x_max, y_max]`
    :param label: a `list` of `str`s that has the class name or label for the object in the
    corresponding box.
    """

    def __init__(self, coords, label: list = None):
        store_attr('coords, label')

    @classmethod
    def listbx(cls, coords, label=None):
        """Classmethod for ListBx. Same as lbx(coords, label)."""
        l = []
        r = []
        for i, c in enumerate(coords):
            assert isinstance(c, (list, tuple, np.ndarray)), f'{inspect.stack()[0][3]} of {__name__}: Expected b of type list/tuple/ndarray, got {type(c)}'
            l_ = c[-1] if len(c) > 4 else '' if label is None else label[i]
            l.append(l_)
            r.append(list(c[:-1]) if len(c) > 4 else c)
        coords = np.array(r)
        return cls(coords, label=l)


class JsonBx:
    """`JsonBx` represents a collection of bounding boxes as a `list` of `dict`s. Internally
     called by `MultiBx` to process coordinates in the format `lists` of `dict`s.

    :param coords: can be nested coordinates of type `list` of `dict`s representing
    multiple boxes. If passing a `dict`, each index of the object should be of the following
    format: `dict` should be in `pascal_voc` format using the keys
    {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1, "label": 'none'} or without "label" key.
    :param label: a `list` of `str`s that has the class name or label for the object in the
    corresponding box. This is useful if records are passed without "label" key.
    """

    def __init__(self, coords, label: list = None):
        store_attr('coords, label')

    @classmethod
    def jsonbx(cls, coords, label=None, keys=None):
        """Classmethod for ListBx. Same as lbx(coords, label)."""
        l = []
        r = []
        for i, c in enumerate(coords):
            assert isinstance(c, dict), f'{inspect.stack()[0][3]} of {__name__}: Expected b of type dict, got {type(c)}'
            if keys is None:
                # Fixes issue #3.
                keys = update_keys(c, default_keys=voc_keys)
            c_ = [c[k] for k in keys]  # read in order
            l_ = c_[-1] if len(c_) > 4 else '' if label is None else label[i]
            l.append(l_)
            r.append(c_[:-1] if len(c_) > 4 else c_)
        coords = np.array(r)
        return cls(coords, label=l)


def get_bx(coords, label=None):
    """Helper function to check and call the correct type of Bx instance.
    :param coords: coordinates in any allowed raw format list/json/dict/ndarray.
    :param label: a `list` of `str`s that has the class name or label for the object in the
    corresponding box.
    :return: an instantialised bounding box.
    """
    if isinstance(coords, np.ndarray):
        coords = np.atleast_2d(coords)
        return mbx(coords, label)
    if isinstance(coords, list):
        if isinstance(coords[0], (np.floating, np.int_, float, int)):
            return bbx(coords, label)
        elif isinstance(coords[0], (np.ndarray, list, dict)):
            return mbx(coords, label)
        elif isinstance(coords[0], tuple):
            return mbx([list(c) for c in coords], label)
    if isinstance(coords, dict):
        return bbx([coords], label)
    if isinstance(coords, tuple):
        return bbx(list(coords), label)
    if isinstance(coords, (MultiBx, ListBx, BaseBx, JsonBx)):
        return coords
    else:
        raise NotImplementedError(f'{inspect.stack()[0][3]} of {__name__}: Got coords={coords} of type {type(coords)}.')


def stack_bxs(b1, b2):
    """Method to stack two BxTypes together. Similar to `__add__` of BxTypes
    but avoids UserWarning.
    :param b1: Bx of class BaseBx, MultiBx, JsonBx, ListBx
    :param b2: Bx of class BaseBx, MultiBx, JsonBx, ListBx
    :return: MultiBx
    """
    if not isinstance(b1, (BaseBx, MultiBx, JsonBx, ListBx)):
        raise TypeError(f'{inspect.stack()[0][3]} of {__name__}: Expected type BaseBx/MultiBx/JsonBx/ListBx, got b1={type(b1)}')
    if not isinstance(b2, (BaseBx, MultiBx, JsonBx, ListBx)):
        raise TypeError(f'{inspect.stack()[0][3]} of {__name__}: Expected type BaseBx/MultiBx/JsonBx/ListBx, got b2={type(b2)}')
    if isinstance(b1, BaseBx):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            return b1 + b2
    return b1 + b2


def add_bxs(b1, b2):
    """Alias of stack_bxs()."""
    return stack_bxs(b1, b2)


def jbx(coords=None, labels=None):
    """Abstraction of the JsonBx class to process `json` records into
    `MultiBx` or `BaseBx` objects exposing many validation methods
    """
    return JsonBx.jsonbx(coords, labels)


def lbx(coords=None, labels=None):
    """Abstraction of the ListBx class to process `list`s of `list`s
    into `MultiBx` or `BaseBx` objects exposing many validation methods
    """
    return ListBx.listbx(coords, labels)


def bbx(coords=None, labels=None):
    """Abstraction of the `BaseBx` class."""
    return BaseBx.basebx(coords, labels)


def mbx(coords=None, labels=None):
    """Abstraction of the `MultiBx` class."""
    return MultiBx.multibox(coords, labels)
