import json
import unittest

import numpy as np

from pybx.basics import mbx, bbx, MultiBx, jbx
from pybx.excepts import BxViolation

np.random.seed(1)

params = {
    "annots_rand_file": '../data/annots_rand.json',
    "annots_iou_file": '../data/annots_iou.json',
    "annots_key_file": '../data/annots_key.json',
    "annots_l": [[50., 70., 120., 100., 'rand1'], [150., 200., 250., 240., 'rand2']],
    "annots_a": np.random.randn(10, 4)
}

results = {
    "mbx_json": (120.0, 'rand2'),
    "mbx_list": (50.0, 'rand1'),
    "mbx_arr": -0.08959797456887511,
    "iou": 0.0425531914893617,
    "xywh": np.array([50.0, 70.0, 70.0, 30.0]),
    "jbx_label": ['person', 4],
}


class BasicsTestCase(unittest.TestCase):
    def test_mbx_json(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        b = mbx(annots)
        r = b.coords[0][2], b.label[1]
        self.assertIsInstance(b, MultiBx, 'b is not MultiBx')
        self.assertEqual(r, results["mbx_json"])

    def test_mbx_list(self):
        annots = params["annots_l"]
        b = mbx(annots)
        r = b.coords[0][2], b.label[1]
        self.assertIsInstance(b, MultiBx, 'b is not MultiBx')
        self.assertEqual(r, results["mbx_json"])

    def test_mbx_array(self):
        annots = params["annots_a"]
        b = mbx(annots)
        r = b.coords.mean()
        self.assertIsInstance(b, MultiBx, 'b is not MultiBx')
        self.assertEqual(r, results["mbx_arr"])

    def test_label_key_jbx(self):
        with open(params["annots_key_file"]) as f:
            annots = json.load(f)
        b_m = jbx(annots)
        self.assertEqual(b_m.label, results["jbx_label"])

    def test_add_bbx(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        b0 = bbx(annots[0])
        b1 = bbx(annots[1])
        b_r = b0 + b1
        b_m = mbx([annots[0], annots[1]])
        self.assertTrue((b_r.coords == b_m.coords).all())

    def test_add_mbx_bbx(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        b_m = mbx([annots[0], annots[1]])
        b1 = bbx(annots[2])
        b_r = b_m + b1
        self.assertTrue((b1.coords == b_r.coords).any(), )

    def test_bbx_warning(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        self.assertRaises(AssertionError, bbx, coords=[annots])

    def test_add_warning(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        b0 = bbx(annots[0])
        b1 = bbx(annots[1])
        self.assertWarns(BxViolation, b0.__add__, other=b1)

    def test_iou(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        b0 = bbx(annots[0])
        b1 = bbx(annots[1])
        b2 = bbx(annots[2])  # intersecting box
        iou = b0.iou(b1)  # calculated iou
        iou_ = b2.area() / (b0.area() + b1.area() - b2.area())
        self.assertEqual(iou, iou_)
        self.assertEqual(iou, results["iou"])

    def test_xywh(self):
        with open(params["annots_rand_file"]) as f:
            annots = json.load(f)
        b = bbx(annots[0])
        self.assertTrue((b.xywh() == results["xywh"]).all(), True)
        self.assertGreaterEqual(b.xywh()[-1], 0)
        self.assertGreaterEqual(b.xywh()[-2], 0)


if __name__ == '__main__':
    unittest.main()
