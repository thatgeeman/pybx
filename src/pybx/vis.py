from collections import defaultdict
from fastcore.basics import store_attr
from matplotlib import pyplot as plt
from matplotlib import patches, patheffects
import numpy as np

from .basics import *
from .sample import get_example


class VisBx:
    def __init__(self, image_sz, **kwargs):
        im, ann, lgt, clr = get_example(image_sz, **kwargs)
        ann = mbx(ann)
        store_attr('im, ann, lgt, clr')

    def show(self, coords, labels=None, color=None, ax=None):
        """
        coords can be a numpy array with labels=(None,labels)
        or a MultiBx, JsonBx, ListBx, BaseBx with labels=None
        """
        if color is not None:
            self.clr.update(color)
        if isinstance(coords, BaseBx):
            coords, labels = coords.make_2d()
        if isinstance(coords, (JsonBx, ListBx)):
            coords, labels = coords.coords, coords.label if coords.label is not None else None
        if isinstance(coords, (list, np.ndarray)):
            # if not multibx, make one so that __add__ below works
            coords = mbx(coords, labels)
        return draw(self.im, self.ann + coords, color=self.clr, logits=self.lgt, ax=ax)


def draw(img: np.ndarray, bbox: list, logits=None, alpha=0.4, **kwargs):
    """
    method to draw an image, box and logits overlayed if passed
    :param img: the image array, expects a numpy array
    :param bbox: list of bounding boxes in json format
    :param logits: activations that should be overlayed from a neural network (no checks)
    :param kwargs: kwargs for draw_boxes()
    :param alpha: same as alpha for matplotlib
    :return: current axis
    """
    ax = draw_boxes(img, bbox, **kwargs)
    if logits is not None:
        img_extent = get_extents(img.shape)
        plt.imshow(logits, alpha=alpha, extent=img_extent)
    return ax


def draw_outline(obj, linewidth: int):
    """
    make outlines around to object edges for visibility in light backgrounds
    :param obj: plt objects like text or rectangle
    :param linewidth: width of the stroke
    :return: plt object
    """
    obj.set_path_effects([patheffects.Stroke(linewidth=linewidth, foreground='black'),
                          patheffects.Normal()])


def draw_text(ax, xy: tuple, label: str, size=12, color='white', xo=0, yo=0):
    """
    write text around boxes
    :param ax: axis object
    :param xy: relative ax coordinates x, y to draw the text
    :param label: label for box
    :param size: font size
    :param yo: y offset for placement of text
    :param xo: x offset for placement of text
    :param color: text color
    :return: ax object
    """
    x, y = xy
    text = ax.text(x + xo, y + yo, label, verticalalignment='top', color=color, fontsize=size)
    draw_outline(text, 1)


def draw_rectangle(ax, coords, color='white'):
    """
    draw a rectangle using matplotlib patch
    :param ax: axis
    :param coords: coordinates in coco format
    :param color: text color
    :return: ax object
    """
    x1, y1, x2, y2 = coords
    w, h = x2 - x1, y2 - y1
    patch = ax.add_patch(patches.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=2))
    draw_outline(patch, 2)


def get_color(color, label=None, default_color='white'):
    """
    if not a string expecting something like a dict
    :param color: dict of key, value pairs where key is label, value is color
    :param label: the label for which color is needed
    :param default_color:
    :return: str that contains color
    """
    if isinstance(color, str):
        return color
    assert label is not None, f'got label={label} to use with color dict {type(color)}'
    colors_d = defaultdict(lambda: default_color)
    colors_d.update(color)
    return colors_d[label]


def get_extents(shape):
    assert len(shape) == 3, f'expected w, h, c = shape, got {shape} with len {len(shape)}'
    w, h, _ = shape
    extent = 0, w, h, 0
    return extent


def draw_boxes(img: np.ndarray, bbox: list, title=None, ax=None, figsize=(5, 4),
               squeeze=False, color='yellow', no_ticks=False, xo=0, yo=0, **kwargs):
    """
    method to draw bounding boxes in an image, can handle multiple bboxes
    :param figsize: sige of figure
    :param img: the image array, expects a numpy array
    :param bbox: list of bounding boxes in json format
    :param title: image title
    :param ax: which axis if already present
    :param squeeze: useful to remove extra axis (if grayscale image channels = 1)
    :param yo: y offset for placement of text
    :param xo: x offset for placement of text
    :param color: text color or dict of colors for each label as a dict
    :param no_ticks: whether to set axis ticks off
    :return: ax with image
    """
    assert isinstance(img, np.ndarray), f'Expected img as np.ndarray, got {type(img)}.'
    if squeeze:
        img = img.squeeze(0)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_tight_layout(True)
    if title is not None:
        ax.set_title(title)
    if no_ticks:
        ax.axis('off')
    ax.imshow(img, cmap='Greys', **kwargs)
    assert isinstance(bbox, (list, BaseBx, MultiBx,
                             np.ndarray)), f'Expected annotations as arrays/list/records/BaseBx/MultiBx, got {type(bbox)}.'
    for b in bbox:
        try:
            x1, y1, x2, y2, label = b.values()
        except ValueError:
            # dict/BaseBx but no label
            x1, y1, x2, y2 = b.values()
            label = ''
        except AttributeError:
            # list without label
            x1, y1, x2, y2 = b
            label = ''
        c = get_color(color, label=label)
        draw_rectangle(ax, coords=(x1, y1, x2, y2), color=c)
        draw_text(ax, xy=(x1, y1), label=label, color=c, xo=xo, yo=yo)
    return ax
