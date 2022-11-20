import unittest
import numpy as np
from pybx.sample import get_example

np.random.seed(1)
params = {
    "feature_szs": [(2, 2), (3, 3), (4, 4)],
    "feature_sz": (2, 2),
    "asp_ratio": 1 / 2.0,
    "image_sz": (10, 10, 3),
    "data_dir": "./data",
}

results = {
    "scaled_ans": (10.546875, 7.03125),
}


class SampleTestCase(unittest.TestCase):
    def test_example(self):
        im, ann, lgts, _ = get_example(
            image_sz=params["image_sz"],
            feature_sz=params["feature_sz"],
            logits=True,
            pth=params["data_dir"],
            load_ann=True,
        )
        self.assertEqual(im.shape, params["image_sz"])
        r = ann[0]["x_max"], ann[1]["y_min"]
        self.assertEqual(r, results["scaled_ans"])
        self.assertEqual(lgts.shape, params["feature_sz"])


if __name__ == "__main__":
    unittest.main()
