import numpy as np
from fastcore.basics import concat, store_attr

from .ops import mul, sub
from .sample import get_example
from .vis import draw


class BaseBx:
    def __init__(self, coords, label='unk'):
        store_attr('coords, label')
        self.w = sub(*coords[::2])
        self.h = sub(*coords[1::2])

    def area(self):
        return abs(mul(self.w, self.h))

    def valid(self):
        return bool(self.area)

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
                label = ['unk'] * coords.shape[0]
            assert coords.shape[0] == len(label), f'wrong shape for coords {coords.shape[0]} and label {len(label)}'
        store_attr('coords, label')

    @classmethod
    def multibox(cls, coords, label: list = None):
        if isinstance(coords, list):
            b = JsonBx.jsonbx(coords)
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
        except IndexError:
            raise StopIteration
        self.index += 1
        return b

    def __add__(self, other):
        if not isinstance(other, (MultiBx, JsonBx)):
            raise TypeError('expected type MultiBx or JsonBx')
        coords = np.vstack([self.coords, other.coords])
        label = self.label + other.label
        return MultiBx(coords, label)


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
            l.append(c_[-1])
            r.append(c_[:-1])
        coords = np.array(r)
        return cls(coords, label=l)


class VisBx:
    def __init__(self, image_sz, **kwargs):
        im, ann, lgt, clr = get_example(image_sz, **kwargs)
        ann = MultiBx.multibox(ann)
        store_attr('im, ann, lgt, clr')

    def show(self, coords: MultiBx, color={'unk': 'white'}, ax=None):
        self.clr.update(color)
        if not isinstance(coords, MultiBx):
            coords = MultiBx.multibox(coords)
        return draw(self.im, self.ann + coords, color=self.clr, logits=self.lgt, ax=ax)
