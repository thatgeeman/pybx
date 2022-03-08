import inspect
from collections import defaultdict

import numpy as np
from fastcore.basics import store_attr
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt

from .basics import *
from .ops import voc_keys
from .sample import get_example, get_given_array


class VisBx:
    """VisBx is used to visualize the bounding boxes.
    The image on of which the bounding boxes are to be drawn can be instantiated with
    `VisBx()` if needed. Calling the `show()` method of the `VisBx()` instance accepts
    bounding box coordinates and labels that are to be shown.
    The boxes can be provided as any of the internal objects (`MultiBx`, `BaseBx`, ...)
    or as any other raw format accepted by the internal objects.

    Displaying image array and annotations object:
        This is the default approach used by VisBx(). If no arguments are passed, a tuple
        denoting the size for random noise `random_img_sz=(100, 100, 1)` is expected.
        Some arguments:
        :param image_arr: image array of shape `(H, W, C)`. If None, it is set to a
            random noise image of `image_sz=(100,100,3)` by default.
        :param annots: annotations is any accepted format (see above).

    Displaying from image and annotations file:
        To load and display the image, set `sample=True`.
        Some argmuments:
        :param ann_fn: annotations file name, default `image.jpg`
        :param img_fn: image file name, default `annots.json`
        :param load_ann: whether to load ann_fn or just the img_fn.
            If False, an empty annotations dict is returned: `dict(zip(voc_keys, [0, 0, 1, 1, '']))`
        :param pth: path to find `ann_fn` and `img_fn`, default `.`
        :param image_sz: size to resize the loaded image a different size (annotations scaled automatically)

    Common parameters:
        :param color: A dict of `color` can be passed to assign specific color to a
            specific `label` in the image: `color = {'frame': 'blue', 'clock': 'green'}`
        :param logits: Logits as `ndarray` that should be overlayed on top of the image
            or `bool` to generate random logits.
        :param feature_sz: Feature size to generate random logits if `logits` is not None.
    """

    def __init__(self, image_arr=None, image_sz=None, sample=False, **kwargs):
        if ('ann_fn' in kwargs) or ('img_fn' in kwargs) or sample:
            assert image_sz is not None, f'{inspect.stack()[0][3]} of {__name__}: Expected image_sz with sample={sample}'
            im, ann, lgt, clr = get_example(image_sz=image_sz, **kwargs)
        else:
            im, ann, lgt, clr = get_given_array(image_arr=image_arr, image_sz=image_sz, **kwargs)
        ann = get_bx(ann)
        store_attr('im, ann, lgt, clr')

    def show(self, coords=None, labels=None, color=None, ax=None, **kwargs):
        """Calling the `show()` method of the `VisBx()` instance accepts
        bounding box coordinates and labels that are to be shown.
        The boxes can be provided as any of the internal objects (`MultiBx`, `BaseBx`, ...)
        or as any other raw format accepted by the internal objects.
        """
        if color is not None:
            self.clr.update(color)
        if coords is None:
            coords = [0, 0, 0, 0]
        coords = get_bx(coords, labels)
        return draw(self.im, self.ann + coords, color=self.clr, logits=self.lgt, ax=ax, **kwargs)


def draw(img: np.ndarray, bbox: list, logits=None, alpha=0.4, **kwargs):
    """Method to draw an image, box and logits overlayed if passed.
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
    """Make outlines around to object edges for visibility in light backgrounds
    :param obj: plt objects like text or rectangle
    :param linewidth: width of the stroke
    :return: plt object
    """
    obj.set_path_effects([patheffects.Stroke(linewidth=linewidth, foreground='black'),
                          patheffects.Normal()])


def draw_text(ax, xy: tuple, label: str, size=12, color='white', xo=0, yo=0):
    """Write text around boxes.
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
    """Draw a rectangle using matplotlib patch.
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
    """Get colors from color dict for a given label. If label=None, return `default_color`.
    :param color: dict of key, value pairs where key is label, value is color
    :param label: the label for which color is needed
    :param default_color:
    :return: str that contains color
    """
    if isinstance(color, str):
        return color
    colors_d = defaultdict(lambda: default_color)
    colors_d.update(color)
    return colors_d[label]


def get_extents(shape):
    """Get extent parameter of the image."""
    assert len(
        shape) == 3, f'{inspect.stack()[0][3]} of {__name__}: Expected w, h, c = shape, got {shape} with len {len(shape)}'
    w, h, _ = shape
    extent = 0, w, h, 0
    return extent


def draw_boxes(img: np.ndarray, bbox: list, title=None, ax=None, figsize=(5, 4),
               color='yellow', no_ticks=False, xo=0, yo=0, **kwargs):
    """Method to draw bounding boxes in an image, can handle multiple bboxes.
    :param figsize: sige of figure
    :param img: the image array, expects a numpy array
    :param bbox: list of bounding boxes in json format
    :param title: image title
    :param ax: which axis if already present
    :param yo: y offset for placement of text
    :param xo: x offset for placement of text
    :param color: text color or dict of colors for each label as a dict
    :param no_ticks: whether to set axis ticks off
    :return: ax with image
    """
    assert isinstance(img,
                      np.ndarray), f'{inspect.stack()[0][3]} of {__name__}: Expected img as np.ndarray, got {type(img)}.'
    assert len(img.shape) == 3, \
        f'{inspect.stack()[0][3]} of {__name__}: Expected img of shape (w, h, c), got {img.shape} with len {len(img.shape)}'
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_tight_layout(True)
    if title is not None:
        ax.set_title(title)
    if no_ticks:
        ax.axis('off')
    ax.imshow(img, cmap='Greys', **kwargs)
    assert isinstance(bbox, (list, BaseBx, MultiBx, np.ndarray)), \
        f'{inspect.stack()[0][3]} of {__name__}: Expected annotations as arrays/list/records/BaseBx/MultiBx, got {type(bbox)}.'

    for b in bbox:
        try:
            x1, y1, x2, y2, label = [b[k] for k in voc_keys]
        except TypeError:
            x1, y1, x2, y2, label = b.values()
        except ValueError:
            if isinstance(b, dict):
                x1, y1, x2, y2 = [b[k] for k in voc_keys[:-1]]
            if isinstance(b, (list, np.ndarray)):
                x1, y1, x2, y2 = b
            label = ''
        c = get_color(color, label=label)
        draw_rectangle(ax, coords=(x1, y1, x2, y2), color=c)
        draw_text(ax, xy=(x1, y1), label=label, color=c, xo=xo, yo=yo)
    return ax
