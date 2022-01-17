import json
import unittest
from pybx import anchor
import numpy as np
from pybx.basics import mbx, MultiBx
from pybx.sample import get_example

np.random.seed(1)

params = {
    "feature_szs": [(2, 2), (3, 3), (4, 4)],
    "asp_ratios": [1 / 2., 1., 2.],
    "feature_sz": (2, 2),
    "asp_ratio": 1 / 2.,
    "image_sz": (10, 10, 3),
    "annots_l": [[50, 70, 100, 120, 'rand1'], [150, 200, 240, 250, 'rand2']],
    "annots_a": np.random.randn(10, 4)
}

results = {
    "test_bx_b": 236.8933982822018,
    "test_bx_l": 'a_2x2_0.5_8',
    "test_bxs_b": 3703.086279536432,
    "test_bxs_l": 'a_4x4_2.0_24',
    "test_basics_mbx_json": (100.0, 'rand2'),
    "test_basics_mbx_list": (50.0, 'rand1'),
    "test_basics_mbx_arr": -0.08959797456887511,
    "scaled_ans": (9.0, 6.0),
}


class MyTestCase(unittest.TestCase):
    def test_anchor_bx(self):
        b, l_ = anchor.bx(params["image_sz"], params["feature_sz"], params["asp_ratio"])
        self.assertIn(results["test_bx_l"], l_, 'label not matching')
        self.assertEqual(b.sum(), results["test_bx_b"], 'sum not matching')  # add assertion here

    def test_anchor_bxs(self):
        b, l_ = anchor.bxs(params["image_sz"], params["feature_szs"], params["asp_ratios"])
        self.assertIn(results["test_bxs_l"], l_, 'label not matching')
        self.assertEqual(b.sum(), results["test_bxs_b"], 'sum not matching')  # add assertion here

    def test_basics_mbx_json(self):
        with open('../data/annots_rand.json') as f:
            annots = json.load(f)
        b = mbx(annots)
        r = b.coords[0][2], b.label[1]
        self.assertIsInstance(b, MultiBx, 'b is not MultiBx')
        self.assertEqual(r, results["test_basics_mbx_json"])

    def test_basics_mbx_list(self):
        annots = params["annots_l"]
        b = mbx(annots)
        r = b.coords[0][2], b.label[1]
        self.assertIsInstance(b, MultiBx, 'b is not MultiBx')
        self.assertEqual(r, results["test_basics_mbx_json"])

    def test_basics_mbx_array(self):
        annots = params["annots_a"]
        b = mbx(annots)
        r = b.coords.mean()
        self.assertIsInstance(b, MultiBx, 'b is not MultiBx')
        self.assertEqual(r, results["test_basics_mbx_arr"])

    def test_sample_ex(self):
        im, ann, _, _ = get_example(params["image_sz"], pth='../data')
        self.assertEqual(im.shape, params["image_sz"])
        r = ann[0]['x_max'], ann[1]['y_min']
        self.assertEqual(r, results['scaled_ans'])


if __name__ == '__main__':
    unittest.main()
