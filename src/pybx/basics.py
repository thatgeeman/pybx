import numpy as np
from fastcore.basics import concat, store_attr

from .anchor import voc_keys
from .ops import mul, sub

__all__ = ['mbx', 'MultiBx', 'BaseBx', 'JsonBx', 'ListBx']


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
        coords_ = coords[::-1]  # reverse
        self.w = sub(*coords_[::2])
        self.h = sub(*coords_[1::2])

    def area(self):
        return abs(mul(self.w, self.h))

    def valid(self):
        # TODO: more validations here
        v_area = bool(self.area())  # False if 0
        # TODO: v_ratio
        v_all = [v_area]
        return False if False in v_all else True

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
            c_ = list(c.values())
            l_ = c_[-1] if len(c_) > 4 else '' if label is None else label[i]
            l.append(l_)
            r.append(c_[:-1] if len(c_) > 4 else c_)
        coords = np.array(r)
        return cls(coords, label=l)


# deprecated
class BxIter:
    def __init__(self, coords: np.ndarray, x_max=-1.0, y_max=-1.0, clip_only=False):
        """
        returns an iterator that validates the coordinates calculated.
        :param coords: ndarray of box coordinates
        :param x_max: max dimension along x
        :param y_max: max dimension along y
        :param clip_only: whether to apply only np.clip with validate
        clip_only cuts boxes that bleed outside limits
        and forgo other validation ops
        """
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        self.coords = coords.clip(0, max(x_max, y_max))
        # clip_only cuts boxes that bleed outside limits
        store_attr('x_max, y_max, clip_only')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            c = self.coords[self.index]
            if not self.clip_only:
                self.validate_edge(c)
        except IndexError:
            raise StopIteration
        self.index += 1
        return c

    def validate_edge(self, c):
        """
        return next only if the x_min and y_min
        # TODO: more tests, check if asp ratio changed as an indicator
        does not flow outside the image, but:
        - while might keep point (1,1,1,1) or line (0,0,1,0) | (0,0,0,1) boxes!
        either maybe undesirable.
        :param c: pass a box
        :return: call for next iterator if conditions not met
        """
        x1, y1 = c[:2]
        if (x1 >= self.x_max) or (y1 >= self.y_max):
            self.index += 1
            return self.__next__()

    def to_array(self, cast_fn=np.asarray):
        """
        return all validated coords as np.ndarray
        :return: array of coordinates, specify get_as torch.tensor for Tensor
        """
        # TODO: fix UserWarning directly casting a numpy to tensor is too slow (torch>10)
        return cast_fn([c for c in self.coords])

    def to_records(self, cast_fn=list):
        """
        return all validated coords as records (list of dicts)
        :return: array of coordinates, specify get_as dict for json
        """
        return cast_fn(dict(zip(voc_keys, [*c, f'a{i}'])) for i, c in enumerate(self.coords))
