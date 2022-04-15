import json
import unittest
import warnings

import numpy as np

from pybx.basics import *
from pybx.vis import VisBx

np.random.seed(1)

params = {
    "data_dir": '../data',
    "annots_file": 'annots_iou.json',
    "annots_iou_file": '../data/annots_iou.json',
    "annots_rand_file": '../data/annots_rand.json',
    "annots_l": [[50., 70., 120., 100., 'rand1'], [150., 200., 250., 240., 'rand2']],
    "annots_1d": np.random.randint(low=1, high=10, size=4),
    "annots_nd": np.random.randint(low=1, high=10, size=(2, 4)),
    "annots_json": [{'label': '', 'x_max': 0, 'x_min': 0, 'y_max': 0, 'y_min': 0}],
    "feature_sz": (2, 2),
    "image_sz": (10, 10, 3),
    "random_im_sz": (10, 10, 3),
    "image_arr": np.random.randint(size=(10, 10, 3), low=0, high=255),
    "image_arr_float": np.random.randn(10, 10, 3),
}


class VisTestCase(unittest.TestCase):
    def __init__(self, args):
        super(VisTestCase, self).__init__(args)
        # use image paths to load image and anns
        self.v1 = VisBx(image_sz=params["image_sz"], feature_sz=params["feature_sz"],
                        logits=True, pth=params["data_dir"], ann_fn=params["annots_file"], sample=True, load_ann=True)

        # use image paths to load image only dont load anns
        self.v2 = VisBx(image_sz=params["image_sz"], pth=params["data_dir"], sample=True, load_ann=False)

        # use image array directly with annots
        self.v3 = VisBx(image_arr=params["image_arr"], annots=params["annots_l"], feature_sz=params["feature_sz"])

        # use image array directly with 1D annots
        self.v4 = VisBx(image_arr=params["image_arr"], annots=params["annots_1d"], feature_sz=params["feature_sz"])

        # use image array directly with ND annots
        self.v5 = VisBx(image_arr=params["image_arr"], annots=params["annots_nd"], feature_sz=params["feature_sz"])

        # use random image array
        self.v6 = VisBx(random_img_sz=params["image_sz"])

        # use logits data with image array
        self.v7 = VisBx(image_arr=params["image_arr"], annots=params["annots_l"], feature_sz=params["feature_sz"],
                        logits=np.random.randn(*params["feature_sz"]))

        # use logits data with image array but single anns
        self.v8 = VisBx(image_arr=params["image_arr"], annots=params["annots_l"][0], feature_sz=params["feature_sz"],
                        logits=np.random.randn(*params["feature_sz"]))

        # use annots json
        self.v9 = VisBx(image_arr=params["image_arr"], annots=params["annots_json"], feature_sz=params["feature_sz"])

        self.vs = [self.v1, self.v2, self.v3, self.v4, self.v5, self.v6, self.v7, self.v8, self.v9]

    def test_vis_bx(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        for v in self.vs:
            self.assertTrue(v.show(annots))

    def test_vis_jsonbx(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        annots = mbx(annots)
        for v in self.vs:
            self.assertTrue(v.show(annots))

    def test_vis_jsonbx_single(self):
        annots = params["annots_json"]
        for v in self.vs:
            self.assertTrue(v.show(annots))

    def test_vis_listbx_single(self):
        annots = bbx(params["annots_l"][0])
        for v in self.vs:
            self.assertTrue(v.show(annots))

    def test_vis_listbx(self):
        annots = mbx(params["annots_l"])
        for v in self.vs:
            self.assertTrue(v.show(annots))

    def test_vis_bbx_list(self):
        b = bbx(params["annots_l"][0])
        self.assertIsInstance(b, BaseBx)
        for v in self.vs:
            self.assertTrue(v.show(b))

    def test_vis_bbx_json(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        b = bbx(annots[0])
        self.assertIsInstance(b, BaseBx)
        for v in self.vs:
            self.assertTrue(v.show(b))

    """
    def test_float_array(self):
        im = params["image_arr_float"]
        ann = params["annots_json"]
        sz = params["image_sz"]
        self.assertRaises(TypeError, VisBx, image_arr=im, image_sz=sz, annots=ann)
    """


if __name__ == '__main__':
    with warnings.catch_warnings:
        warnings.filterwarnings('ignore')
        unittest.main()
