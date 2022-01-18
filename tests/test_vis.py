import json
import unittest

from pybx.basics import bbx, BaseBx
from pybx.vis import VisBx

params = {
    "data_dir": '../data',
    "annots_iou_file": '../data/annots_iou.json',
    "annots_rand_file": '../data/annots_rand.json',
    "annots_l": [[50., 70., 120., 100., 'rand1'], [150., 200., 250., 240., 'rand2']],
    "feature_sz": (2, 2),
    "image_sz": (10, 10, 3),
}


class VisTestCase(unittest.TestCase):
    def __init__(self, args):
        super(VisTestCase, self).__init__(args)
        self.v = VisBx(params["image_sz"], feature_sz=params["feature_sz"], logits=True, pth=params["data_dir"])

    def test_vis_bx(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        ax = self.v.show(annots)
        self.assertTrue(ax)

    def test_vis_jsonbx(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        ax = self.v.show(annots)
        self.assertTrue(ax)

    def test_vis_listbx(self):
        ax = self.v.show(params["annots_l"])
        self.assertTrue(ax)

    def test_vis_bbx_list(self):
        b = bbx(params["annots_l"][0])
        ax = self.v.show(b)
        self.assertTrue(ax)
        self.assertIsInstance(b, BaseBx)

    def test_vis_bbx_json(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        b = bbx(annots[0])
        ax = self.v.show(b)
        self.assertTrue(ax)
        self.assertIsInstance(b, BaseBx)


if __name__ == '__main__':
    unittest.main()
