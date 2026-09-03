import unittest

import numpy as np

from pybx.utils import validate_boxes


class UtilsTestCase(unittest.TestCase):
    def test_validate_boxes_clips_landscape_coordinates(self):
        coords = np.array([[-10, -5, 90, 45]], dtype=float)
        result = validate_boxes(coords, (100, 50), (2, 2), min_visibility=0)
        self.assertEqual(result, [[0, 0, 90, 45]])

    def test_validate_boxes_clips_portrait_coordinates(self):
        coords = np.array([[-10, -5, 45, 90]], dtype=float)
        result = validate_boxes(coords, (50, 100), (2, 2), min_visibility=0)
        self.assertEqual(result, [[0, 0, 45, 90]])

    def test_validate_boxes_clips_maxima_to_the_correct_axis(self):
        coords = np.array([[10, 5, 120, 80]], dtype=float)
        result = validate_boxes(coords, (100, 50), (2, 2), min_visibility=0)
        self.assertEqual(result, [[10, 5, 100, 50]])


if __name__ == "__main__":
    unittest.main()
