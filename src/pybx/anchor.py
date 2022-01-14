import math

import numpy as np
from operator import __mul__

from fastcore.basics import store_attr

from show import draw
from utils import allowed_ops, voc_keys, get_op


def get_edges(image_sz: tuple, feature_sz: tuple, op='noop'):
    """
    generate offsetted top (x_min, y_min) or bottom edges (x_max, y_max)
    coordinates of a given feature size based on op.
    if op is noop, gets the top edges.
    if op is sub, gets the bottom edges.
    :param op: operation for calculating edges, either 'add' 'subtract' 'noop'
    :param image_sz: tuple of (width, height, channels) of an image
    :param feature_sz: tuple of (width, height) of a channel
    :return: offsetted edges of each feature
    """
    assert image_sz[-1] < image_sz[0], f'expected {image_sz[-1]} < {image_sz[0]}={image_sz[1]}'
    assert len(image_sz) == 3, f'expected image_sz of len 3, got {len(image_sz)}'
    assert op in allowed_ops, f'operator not in allowed operations: {allowed_ops}'
    w, h, _ = image_sz
    nx, ny = feature_sz
    diag_edge_ofs = w / nx, h / ny
    op = get_op(op)
    x_ = op(np.linspace(0, w, nx + 1), diag_edge_ofs[0])
    y_ = op(np.linspace(0, h, ny + 1), diag_edge_ofs[1])
    mesh = np.meshgrid(x_, y_)
    edges = np.stack([m.flatten() for m in mesh], axis=-1)
    return edges


def calc_coords(image_sz: tuple, feature_sz: tuple, asp_ratio: float, show=False, validate=True, clip_only=False):
    """
    calculate anchor box coords given an image size and feature size for a single aspect ratio
    :param image_sz: tuple of (width, height) of an image
    :param feature_sz: tuple of (width, height) of a channel
    :param asp_ratio: aspect ratio (width:height)
    :param show: whether to display the generated anchors
    :param validate: fix the boxes that are bleeding out of the image
    :param clip_only: whether to apply only np.clip with validate
    :return: anchor box coordinates in [pascal_voc] format
    """
    n_boxes = __mul__(*feature_sz)
    top_edges = get_edges(image_sz, feature_sz, op='noop')
    bot_edge = get_edges(image_sz, feature_sz, op='add')
    coords = np.hstack([top_edges, bot_edge])  # raw coords
    coords_wh = (coords[:, 2:] - coords[:, :2])  # w -> xmax-xmin, h -> ymax-ymin
    coords_center = (coords[:, 2:] - coords_wh / 2)  # xmax-w/2, ymax-h/2
    # scale the dimension of width and height with asp ratios
    _w = coords_wh[:, 0] * math.sqrt(asp_ratio)
    _h = coords_wh[:, 1] / math.sqrt(asp_ratio)
    coords_asp_wh = np.stack([_w, _h], -1)
    # TODO: given the center, any format for the anchors can be obtained
    coords_asp = np.hstack([coords_center - coords_asp_wh / 2, coords_center + coords_asp_wh / 2])
    if validate:
        # TODO: validate with box iou of coords and given bounding box
        coords_iter = CoordClass(coords_asp, x_max=image_sz[0], y_max=image_sz[1], clip_only=clip_only)
        coords_asp = coords_iter.to_array()
    if show:
        _ = draw(coords_asp)
    return coords_asp


class CoordClass:
    def __init__(self, coords: np.ndarray, x_max=-1.0, y_max=-1.0, clip_only=True):
        """
        returns an iterator that validates the coordinates calculated.
        :param coords: ndarray of box coordinates
        :param x_max: max dimension along x
        :param y_max: max dimension along y
        :param clip_only: whether to apply only np.clip with validate
        """
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        self.coords = coords.clip(0, max(x_max, y_max))
        store_attr('x_max, y_max, clip_only')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        """
        :return: one anchor box at each iteration
        """
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
        does not flow outside the image, can remove some nice boxes!
        while might keep point (1,1,1,1) boxes! either maybe undesirable.
        :param c: pass a box
        :return:
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
        # TODO: UserWarning directly casting a numpy to tensor is too slow (torch>10)
        return cast_fn([c for c in self.coords])

    def to_records(self, cast_fn=list):
        """
        return all validated coords as records (list of dicts)
        :return: array of coordinates, specify get_as dict for json
        """
        return cast_fn(dict(zip(voc_keys, [*c, f'a{i}'])) for i, c in enumerate(self.coords))
