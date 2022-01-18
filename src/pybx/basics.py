import numpy as np
from fastcore.basics import concat, store_attr

from .ops import mul, sub, intersection_box, make_array, NoIntersection, voc_keys

__all__ = ['bbx', 'mbx', 'MultiBx', 'BaseBx', 'JsonBx', 'ListBx']


def bbx(coords=None, labels=None):
    """
    interface to the BaseBx class and all of its attributes
    MultiBx wraps the coordinates and labels exposing many validation methods
    :param coords: coordinates in list/array/json format
    :param labels: labels in list format or keep intentionally None (also None for json)
    :return: BaseBx object
    """
    return BaseBx.basebx(coords, labels)


def mbx(coords=None, labels=None):
    """
    interface to the MultiBx class and all of its attributes
    MultiBx wraps the coordinates and labels exposing many validation methods
    :param coords: coordinates in list/array/json format
    :param labels: labels in list format or keep intentionally None (also None for json)
    :return: MultiBx object
    """
    return MultiBx.multibox(coords, labels)


class BaseBx:
    def __init__(self, coords, label=''):
        store_attr('coords, label')
        self.w = sub(*coords[::2][::-1])
        self.h = sub(*coords[1::2][::-1])

    def area(self):
        return abs(mul(self.w, self.h))

    def iou(self, other):
        if not isinstance(other, BaseBx):
            other = BaseBx.basebx(other)
        if self.valid():
            try:
                int_box = BaseBx.basebx(intersection_box(self.coords, other.coords))
            except NoIntersection:
                return 0.0
            int_area = int_box.area()
            union_area = other.area() + self.area() - int_area
            return int_area / union_area
        return 0.0

    def valid(self):
        v_area = bool(self.area())  # False if 0
        v_all = np.array([v_area])
        return True if v_all.all() else False

    def values(self):
        return [*self.coords, self.label]

    def xywh(self):
        return np.asarray(concat([self.coords[:2], self.w, self.h]))

    def __len__(self):
        return self.coords.shape[0] + 1

    def make_2d(self):
        coords = np.atleast_2d(self.coords)
        labels = [self.label]
        return coords, labels

    @classmethod
    def basebx(cls, coords, label: list = None):
        if not isinstance(coords, np.ndarray):
            try:
                coords, label = make_array(coords)
            except ValueError:
                coords = make_array(coords)
        return cls(coords, label)


class MultiBx:
    def __init__(self, coords, label: list = None):
        if isinstance(coords, np.ndarray):
            if label is None:
                label = [''] * coords.shape[0]
            assert coords.shape[0] == len(label), f'wrong shape for coords {coords.shape[0]} and label {len(label)}'
        store_attr('coords, label')

    @classmethod
    def multibox(cls, coords, label: list = None):
        if isinstance(coords, list):
            try:
                b = JsonBx.jsonbx(coords, label)
            except AssertionError:
                # if dict assertion fails
                b = ListBx.listbx(coords, label)
            return cls(b.coords, b.label)
        return cls(coords, label)

    def __getitem__(self, idx):
        return BaseBx(self.coords[idx], label=self.label[idx])

    def __len__(self):
        return len(self.coords)

    def shape(self):
        return self.coords.shape

    def __iter__(self):
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
        if not isinstance(other, (MultiBx, JsonBx, ListBx)):
            raise TypeError('expected type MultiBx/JsonBx/ListBx')
        coords = np.vstack([self.coords, other.coords])
        label = self.label + other.label
        return MultiBx(coords, label)


class ListBx:
    def __init__(self, coords, label: list = None):
        store_attr('coords, label')

    @classmethod
    def listbx(cls, coords, label=None):
        l = []
        r = []
        for i, c in enumerate(coords):
            assert isinstance(c, (list, np.ndarray)), f'expected b of type list/ndarray, got {type(c)}'
            l_ = c[-1] if len(c) > 4 else '' if label is None else label[i]
            l.append(l_)
            r.append(c[:-1] if len(c) > 4 else c)
        coords = np.array(r)
        return cls(coords, label=l)


class JsonBx:
    def __init__(self, coords, label: list = None):
        store_attr('coords, label')

    @classmethod
    def jsonbx(cls, coords, label=None):
        l = []
        r = []
        for i, c in enumerate(coords):
            assert isinstance(c, dict), f'expected b of type dict, got {type(c)}'
            c_ = [c[k] for k in voc_keys]  # read in order
            l_ = c_[-1] if len(c_) > 4 else '' if label is None else label[i]
            l.append(l_)
            r.append(c_[:-1] if len(c_) > 4 else c_)
        coords = np.array(r)
        return cls(coords, label=l)
