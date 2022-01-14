import json
import os

import numpy as np
from PIL import Image


def _get_scaled_annots(annots: list, new_sz: tuple, ann_im_sz=(300, 300, 3)):
    """
    scales the bounding boxes with change in the image size
    :param annots: bounding boxes in records format
    :param new_sz: new size of image (after transform)
    :param ann_im_sz: original size of image
    :return:
    """
    scaled = []
    for annot in annots:
        d = {}
        assert isinstance(annot, dict), f'expected annots of type dict, got {type(annots)}'
        for k, v in annot.items():
            if k.startswith('x'):
                v_ = new_sz[0] * v / ann_im_sz[0]
            elif k.startswith('y'):
                v_ = new_sz[1] * v / ann_im_sz[1]
            else:
                # dont destroy other keys
                v_ = v
            d.update({k: v_})
        scaled.append(d)
    return scaled


def _get_example(image_sz: tuple, feature_sz: tuple, ann_im_sz=(300, 300, 3),
                 pth='.', img_fn='image.jpg', ann_fn='annots.json'):
    """
    get an example image from the pth given for some image size for a feature size
    :param image_sz: required image size (will resize the original image)
    :param feature_sz: feature size of
    :param pth: path to find image
    :param ann_im_sz: original image size
    :return:
    """
    assert os.path.exists(os.path.join(pth, img_fn)), f'{pth} has no {img_fn}'
    assert os.path.exists(os.path.join(pth, 'annots.json')), f'{pth} has no {ann_fn}'
    assert len(image_sz) == 3, f'expected w, h, c in image_sz, got {image_sz} with len {len(image_sz)}'
    im = Image.open(os.path.join(pth, img_fn)).convert('RGB').resize(list(image_sz[:2]))
    im_array = np.asarray(im)
    logits = _get_feature(feature_sz)
    with open(os.path.join(pth, ann_fn)) as f:
        annots = json.load(f)  # annots for 300x300 image
    annots = _get_scaled_annots(annots, image_sz, ann_im_sz=ann_im_sz)
    color = {'frame': 'blue', 'clock': 'green'}
    return im_array, annots, logits, color


def _get_feature(feature_sz: tuple):
    """
    get fake features for some layer in decoder of size feature_sz
    :param feature_sz: size of random features
    :return:
    """
    return np.random.randn(*feature_sz)
