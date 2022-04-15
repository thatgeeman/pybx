import unittest

import numpy as np

from pybx import anchor

np.random.seed(1)

params = {
    "feature_szs": [(2, 2), (3, 3), (4, 4)],
    "asp_ratios": [1 / 2., 1., 2.],
    "feature_sz": (2, 2),
    "asp_ratio": 1 / 2.,
    "image_sz": (10, 10, 3),
    "data_dir": './data',
}

results = {
    "bx_b": 80.0,
    "bx_l": 'a_2x2_0.5_3',
    "bxs_b": 1740.0,
    "bxs_l": 'a_3x3_0.5_2',
    "scaled_ans": (9.0, 6.0),
}


class AnchorTestCase(unittest.TestCase):
    def test_bx(self):
        b, l_ = anchor.bx(params["image_sz"], params["feature_sz"], params["asp_ratio"])
        self.assertIn(results["bx_l"], l_, 'label not matching')
        self.assertEqual(len(b), len(l_))
        self.assertEqual(b.sum(), results["bx_b"], 'sum not matching')  # add assertion here

    def test_bx_dtype(self):
        b = anchor.bx(params["image_sz"], params["feature_sz"], params["asp_ratio"], named=False)
        self.assertIsInstance(b, np.ndarray, 'box type is not ndarray')
        self.assertIsInstance(b[0], np.ndarray, 'box item type is not array')

    def test_bx_dtype_named(self):
        b, l_ = anchor.bx(params["image_sz"], params["feature_sz"], params["asp_ratio"], named=True)
        self.assertIsInstance(b, np.ndarray, 'box type is not ndarray')
        self.assertIsInstance(b[0], np.ndarray, 'box item type is not array')
        self.assertIsInstance(l_, list, 'label type is not list')
        self.assertIsInstance(l_[0], str, 'label item type is not str')

    def test_bxs(self):
        b, l_ = anchor.bxs(params["image_sz"], params["feature_szs"], params["asp_ratios"])
        self.assertIn(results["bxs_l"], l_, 'label not matching')
        self.assertEqual(len(b), len(l_))
        self.assertEqual(b.sum(), results["bxs_b"], 'sum not matching')  # add assertion here


if __name__ == '__main__':
    unittest.main()
