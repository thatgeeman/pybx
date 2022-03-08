import inspect
import json
import os

import numpy as np
from PIL import Image
from fastcore.test import nequals

__all__ = ['get_example', 'get_given_array']

from .ops import voc_keys

np.random.seed(1)


def get_example(image_sz: tuple, **kwargs):
    """Get an example image from the pth given for some image size for a feature size
    :param image_sz: required image size (will resize the original image)
    :return: reference to protected _get_example()
    """
    return _get_example(image_sz, **kwargs)


def get_given_array(image_arr, **kwargs):
    """Get the image_array setup for visualisation.
    :param image_arr: image nparray
    :return: reference to protected _get_given_array()
    """
    return _get_given_array(image_arr, **kwargs)


def _scale_annots_dict(annot, new_sz, ann_im_sz):
    """Scale annotations to the new_sz, provided the original ann_im_sz.
    :param annot: bounding box in dict format
    :param new_sz: new size of image (after linear transforms like resize)
    :param ann_im_sz: original size of image for which the bounding boxes were given.
    :return:
    """
    d = {}
    for k, v in annot.items():
        if k.startswith('x'):
            v_ = new_sz[0] * v / ann_im_sz[0]
        elif k.startswith('y'):
            v_ = new_sz[1] * v / ann_im_sz[1]
        else:
            # don't destroy other keys
            v_ = v
        d.update({k: v_})
    return d


def _scale_annots_list(annot, new_sz, ann_im_sz):
    """Convert annot to dict format before calling _scale_annots_dict().
    :param annot: bounding box in list format
    :param new_sz: new size of image (after linear transforms like resize)
    :param ann_im_sz: original size of image for which the bounding boxes were given.
    :return:
    """
    annot = dict(zip(voc_keys, annot)) if len(annot) == 5 else dict(zip(voc_keys[:-1], annot))
    return _scale_annots_dict(annot, new_sz, ann_im_sz)


def _get_scaled_annots(annots: list, new_sz: tuple, ann_im_sz=(300, 300, 3)):
    """Scales the bounding boxes with change in the image size.
    :param annots: bounding boxes in records format
    :param new_sz: new size of image (after linear transforms like resize)
    :param ann_im_sz: original size of image for which the bounding boxes were given.
    :return:
    """
    scaled = []
    for annot in annots:
        if isinstance(annot, dict):
            d = _scale_annots_dict(annot, new_sz, ann_im_sz)
        elif isinstance(annot, list):
            d = _scale_annots_list(annot, new_sz, ann_im_sz)
        else:
            raise NotImplementedError(
                f'{inspect.stack()[0][3]} of {__name__}: Expected annot of type dict, got {type(annot)}')
        scaled.append(d)
    return scaled


def _get_example(image_sz: tuple = None, feature_sz: tuple = None, pth='.', img_fn='image.jpg',
                 load_ann=True, ann_fn='annots.json', logits=None, color: dict = {}):
    """Get an example image from the pth given for some image size for a feature size.
    :param image_sz: size to resize the loaded image a different size (annotations scaled automatically)
    :param feature_sz: Feature size to generate random logits if `logits` is not None.
    :param pth: path to find `ann_fn` and `img_fn`, default `.`
    :param img_fn: image file name, default `annots.json`
    :param load_ann: whether to load ann_fn or just the img_fn.
            If False, an empty annotations dict is returned: `[dict(zip(voc_keys, [0, 0, 1, 1, '']))]`
    :param ann_fn: annotations file name, default `image.jpg`
    :param logits: activations that should be overlayed from a neural network (no checks)
    :param color: A dict of `color` can be passed to assign specific color to a
            specific `label` in the image: `color = {'frame': 'blue', 'clock': 'green'}`
    :returns: image_arr, annots, logits, color
    """
    assert os.path.exists(os.path.join(pth, img_fn)), f'{pth} has no {img_fn}'
    assert len(image_sz) == 3, f'{inspect.stack()[0][3]} of {__name__}: \
    Expected w, h, c in image_sz, got {image_sz} with len {len(image_sz)}'

    im = Image.open(os.path.join(pth, img_fn)).convert('RGB')
    image_arr = np.asarray(im)
    image_sz = image_arr.shape if image_sz is None else image_sz  # size to reshape into
    ann_im_sz = image_arr.shape  # original size

    annots = [dict(zip(voc_keys, [0, 0, 1, 1, '']))]  # default values
    if load_ann:
        assert ann_fn is not None, f'{inspect.stack()[0][3]} of {__name__}: \
        got ann_fn={ann_fn} with show_ann={load_ann}'

        assert os.path.exists(os.path.join(pth, ann_fn)), f'{pth} has no {ann_fn}'
        with open(os.path.join(pth, ann_fn)) as f:
            annots = json.load(f)  # annots for 300x300 image

    if nequals(ann_im_sz, image_sz): # if not equal, returns True
        image_arr = _get_resized(im, image_sz)
        annots = _get_scaled_annots(annots, image_sz, ann_im_sz=ann_im_sz)

    assert isinstance(annots, list), f'{inspect.stack()[0][3]} of {__name__}: \
    Expected annots should be list of list/dict, got {annots} of type {type(annots)}'

    if logits is not None:
        # if ndarray/detached-tensor, use logits values
        if not hasattr(logits, 'shape'):
            assert feature_sz is not None, f'{inspect.stack()[0][3]} of {__name__}: \
            Expected feature_sz to generate fake-logits'

            logits = _get_feature(feature_sz)
    color = {'frame': 'blue', 'clock': 'green'} if not color else color
    return image_arr, annots, logits, color


def _get_resized(im, image_sz):
    """Resize `im` to `image_sz` using PIL."""
    # print(f'image_arr @ {__name__} is {image_arr.shape}')
    new_sz = list(image_sz[:2])
    im = im.resize(size=new_sz)
    # print(f'im @ {__name__} is {im}')
    return np.asarray(im)


def _get_random_im(image_sz):
    """Returns a randomly generated 8-bit image."""
    return np.random.randint(size=image_sz, low=0, high=255, dtype=np.uint8)


def _get_given_array(image_arr: np.ndarray = None, annots: list = None, image_sz=None, random_img_sz=None,
                     logits=None, feature_sz: tuple = None, color: dict = {}):
    """To display image array and annotations object. This is the default approach used by vis.VisBx
    :param image_arr: image array of shape `(H, W, C)`. If None, it is set to a
            random noise image of `image_sz=(100,100,3)` by default.
    :param annots: annotations is any accepted format. The boxes can be provided as any of the internal
        objects (`MultiBx`, `BaseBx`, ...) or as any other raw format accepted by the internal objects.
    :param image_sz: Size of the random image to be generated if `image_arr` is None.
        `v = vis.VisBx()` has all params set to None. If None, a random noise of `image_sz=(100, 100, 1)` is used.
        This random noise is the default image. If passed along with `image_arr`, then `image_arr` is reshaped to
        `image_sz` and annotations are scaled.
    :param random_img_sz: Size of the image for random noise generation, default (100, 100, 3)
    :param logits: Logits as `ndarray` that should be overlayed on top of the image
            or `bool` to generate random logits.
    :param feature_sz: Feature size to generate random logits if `logits` is not None.
    :param color: A dict of `color` can be passed to assign specific color to a
            specific `label` in the image: `color = {'frame': 'blue', 'clock': 'green'}`
    :returns: image_arr, annots, logits, color
    """
    if image_arr is None:
        assert random_img_sz is not None, f'With image_arr={image_arr}, expected random_img_sz param to generate random noise.'
        image_arr = _get_random_im(random_img_sz)
    ann_im_sz = image_arr.shape  # size for which annotations are given
    image_sz = image_arr.shape if image_sz is None else image_sz  # if resize is needed
    if nequals(ann_im_sz, image_sz):  # if not equal, returns True
        # print(f'The input size is {image_sz}, anchors calculated for {ann_im_sz}')
        im = Image.fromarray(image_arr.astype(np.uint8))
        image_arr = _get_resized(im, image_sz)
        annots = _get_scaled_annots(annots, image_sz, ann_im_sz=ann_im_sz)
    if logits is not None:
        # if ndarray/detached-tensor, use logits values
        if not hasattr(logits, 'shape'):
            assert feature_sz is not None, f'{inspect.stack()[0][3]} of {__name__}: \
            Expected feature_sz to generate fake-logits'
            logits = _get_feature(feature_sz)
    if annots is None:
        annots = [{k: 0 if k != 'label' else '' for k in voc_keys}]
    return image_arr, annots, logits, color if not color else color


def _get_feature(feature_sz: tuple):
    """Get fake features for some layer in decoder of size feature_sz
    :param feature_sz: size of random features
    :return:
    """
    return np.random.randn(*feature_sz)
