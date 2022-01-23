import json
import unittest

import numpy as np

from pybx import ops
from pybx.excepts import NoIntersection

np.random.seed(1)

params = {
    "data_dir": '../data',
    "annots_iou_file": '../data/annots_iou.json',
    "annots_rand_file": '../data/annots_rand.json',
}

results = {
    "add": 18,
    "sub": 2,
    "noop": 10,
    "mul": 80,
    "namedidx": 'a3'
}


class OpsTestCase(unittest.TestCase):
    def test_get_op(self):
        for o in ops.__ops__:
            o_ = ops.get_op(o)
            self.assertEqual(o_(10, 8), results[o], 'op results not matching')

    def test_make_array(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        array, label = ops.make_array(annots[0])
        self.assertIsInstance(array, np.ndarray)
        self.assertIsInstance(label, list)

    def test_named_idx(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        array, label = ops.make_array(annots[1])
        namedidx = ops.named_idx(array, 'a')
        self.assertEqual(namedidx[-1], results["namedidx"])

    def test_intersection_box(self):
        with open(params["annots_iou_file"]) as f:
            annots = json.load(f)
        a0, _ = ops.make_array(annots[0])
        a1, _ = ops.make_array(annots[1])
        a2, _ = ops.make_array(annots[2])
        int_box_array = ops.intersection_box(a1, a2)
        self.assertTrue((a2 == int_box_array).sum())

    def test_intersection_box_noint(self):
        with open(params["annots_iou_file"]) as f:
            annots0 = json.load(f)
        with open(params["annots_rand_file"]) as f:
            annots1 = json.load(f)
        a1, _ = ops.make_array(annots0[0])
        a2, _ = ops.make_array(annots1[1])
        self.assertRaises(NoIntersection, ops.intersection_box, b1=a1, b2=a2)


if __name__ == '__main__':
    unittest.main()
