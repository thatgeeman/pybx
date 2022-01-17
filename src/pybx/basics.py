import numpy as np
from fastcore.basics import concat, store_attr

from .anchor import voc_keys
from .ops import mul, sub
from .sample import get_example
from .vis import draw


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
        val_area = bool(self.area())
        return False if False in [val_area] else True

    def values(self):
        return [*self.coords, self.label]

    def xywh(self):
        return np.asarray(concat([self.coords[:2], self.w, self.h]))

    def __len__(self):
        return self.coords.shape[0] + 1


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
                b = JsonBx.jsonbx(coords)
            except:
                b = ListBx.listbx(coords)
            return cls(b.coords, b.label)
        return cls(coords, label)

    def __getitem__(self, idx):
        return BaseBx(self.coords[idx], label=self.label[idx])

    def __len__(self):
        return len(self.coords)

    @property
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
                self.__next__()
        except IndexError:
            raise StopIteration
        self.index += 1
        return b

    def __add__(self, other):
        if not isinstance(other, (MultiBx, JsonBx, ListBx)):
            raise TypeError('expected type MultiBx or JsonBx')
        coords = np.vstack([self.coords, other.coords])
        label = self.label + other.label
        return MultiBx(coords, label)


class ListBx:
    def __init__(self, coords, label: list = [None]):
        store_attr('coords, label')

    @classmethod
    def listbx(cls, coords):
        l = []
        r = []
        for c in coords:
            assert isinstance(c, list), f'expected b of type list, got {type(c)}'
            l_ = c[-1] if len(c) > 4 else ''
            l.append(l_)
            r.append(c)
        coords = np.array(r)
        return cls(coords, label=l)


class JsonBx:
    def __init__(self, coords, label: list = [None]):
        store_attr('coords, label')

    @classmethod
    def jsonbx(cls, coords):
        l = []
        r = []
        for c in coords:
            assert isinstance(c, dict), f'expected b of type dict, got {type(c)}'
            c_ = list(c.values())
            l_ = c_[-1] if len(c_) > 4 else ''
            l.append(l_)
            r.append(c_[:-1])
        coords = np.array(r)
        return cls(coords, label=l)


class VisBx:
    def __init__(self, image_sz, **kwargs):
        im, ann, lgt, clr = get_example(image_sz, **kwargs)
        ann = MultiBx.multibox(ann)
        store_attr('im, ann, lgt, clr')

    def show(self, coords: MultiBx, labels=None, color=None, ax=None):
        if color is not None:
            self.clr.update(color)
        if not isinstance(coords, MultiBx):
            coords = MultiBx.multibox(coords, labels)
        return draw(self.im, self.ann + coords, color=self.clr, logits=self.lgt, ax=ax)


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
