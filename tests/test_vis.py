import json
import unittest

import matplotlib.pyplot
import numpy as np

from pybx.basics import bbx, BaseBx, mbx
from pybx.vis import VisBx

np.random.seed(1)

params = {
    "data_dir": '../data',
    "annots_iou_file": '../data/annots_iou.json',
    "annots_rand_file": '../data/annots_rand.json',
    "annots_l": [[50., 70., 120., 100., 'rand1'], [150., 200., 250., 240., 'rand2']],
    "annots_json": [{'label': '', 'x_max': 0, 'x_min': 0, 'y_max': 0, 'y_min': 0}],
    "feature_sz": (2, 2),
    "image_sz": (10, 10, 3),
    "image_arr": np.random.randint(size=(10, 10, 3), low=0, high=255)
}


class VisTestCase(unittest.TestCase):
    def __init__(self, args):
        super(VisTestCase, self).__init__(args)
        # use image paths to load image
        self.v1 = VisBx(image_sz=params["image_sz"], feature_sz=params["feature_sz"],
                        logits=True, pth=params["data_dir"], sample=True)
        # use image array directly with annots
        self.v2 = VisBx(image_arr=params["image_arr"], annots=params["annots_l"], feature_sz=params["feature_sz"])

        # use random image array
        self.v3 = VisBx()

        # use logits data with image array
        self.v4 = VisBx(image_arr=params["image_arr"], annots=params["annots_l"], feature_sz=params["feature_sz"],
                        logits=np.random.randn(*params["feature_sz"]))

        # use annots json
        self.v5 = VisBx(image_arr=params["image_arr"], annots=params["annots_json"], feature_sz=params["feature_sz"])

        self.vs = [self.v1, self.v2, self.v3, self.v4, self.v5]

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


if __name__ == '__main__':
    unittest.main()
