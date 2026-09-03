import unittest

import numpy as np

from pybx import anchor
from pybx.basics import ITER_TYPES, ITER_TYPES_EXTRA, ITER_TYPES_TUPLE

np.random.seed(1)

params = {
    "feature_szs": [(2, 2), (3, 3), (4, 4)],
    "asp_ratios": [1 / 2.0, 1.0, 2.0],
    "feature_sz": (2, 2),
    "asp_ratio": 1 / 2.0,
    "image_sz": (10, 10),
    "data_dir": "./data",
}

results = {
    "bx_b": 128,
    "bx_l": "a_2x2_0.5_3",
    "bxs_b": 2002,
    "bxs_l": "a_3x3_0.5_2",
    "scaled_ans": (9.0, 6.0),
}


class AnchorTestCase(unittest.TestCase):
    def test_bx(self):
        b, l_ = anchor.bx(params["image_sz"], params["feature_sz"], params["asp_ratio"])
        self.assertIn(results["bx_l"], l_, "label not matching")
        self.assertEqual(len(b), len(l_))
        self.assertEqual(
            np.sum(list(b)), results["bx_b"], "sum not matching"
        )  # add assertion here

    def test_bx_dtype(self):
        b = anchor.bx(
            params["image_sz"], params["feature_sz"], params["asp_ratio"], named=False
        )
        self.assertIsInstance(b, ITER_TYPES)
        self.assertIsInstance(b[0], ITER_TYPES)

    def test_bx_dtype_named(self):
        b, l_ = anchor.bx(
            params["image_sz"], params["feature_sz"], params["asp_ratio"], named=True
        )
        self.assertIsInstance(b, ITER_TYPES)
        self.assertIsInstance(b[0], ITER_TYPES)
        self.assertIsInstance(l_, ITER_TYPES)
        self.assertIsInstance(l_[0], str)

    def test_bxs(self):
        b, l_ = anchor.bxs(
            params["image_sz"], params["feature_szs"], params["asp_ratios"]
        )
        self.assertIn(results["bxs_l"], l_, "label not matching")
        self.assertEqual(len(b), len(l_))
        self.assertEqual(
            b.sum(), results["bxs_b"], "sum not matching"
        )  # add assertion here

    def test_get_gt_offsets_accepts_dict_annotation(self):
        annotation = {
            "x_min": 0,
            "y_min": 0,
            "x_max": 2,
            "y_max": 2,
            "label": "cat",
        }
        offsets, labels = anchor.get_gt_offsets(
            annotation, [[0, 0, 2, 2]], update_labels=True
        )
        np.testing.assert_array_equal(offsets, np.zeros((1, 4)))
        self.assertEqual(labels, ["cat"])

    def test_matching_keeps_objects_with_the_same_label_separate(self):
        matches, ious, masks = anchor.get_gt_max_iou(
            [[0, 0, 10, 10, "cat"], [20, 20, 30, 30, "cat"]],
            [[0, 0, 10, 10], [20, 20, 30, 30]],
            box_ids=["cat-left", "cat-right"],
            return_ious=True,
            return_masks=True,
        )

        self.assertEqual(set(matches), {"cat-left", "cat-right"})
        self.assertEqual(matches["cat-left"].coords, [[0, 0, 10, 10]])
        self.assertEqual(matches["cat-right"].coords, [[20, 20, 30, 30]])
        self.assertEqual(matches["cat-left"].label, ["cat"])
        self.assertEqual(ious, {"cat-left": [1.0], "cat-right": [1.0]})
        self.assertEqual(masks, {"cat-left": [True, False], "cat-right": [False, True]})

    def test_matching_uses_input_positions_as_default_box_ids(self):
        matches, _, masks = anchor.get_gt_thresh_iou(
            [[0, 0, 10, 10, "cat"], [0, 0, 10, 10, "cat"]],
            [[0, 0, 10, 10]],
            iou_thresh=0.5,
            return_masks=True,
        )

        self.assertEqual(set(matches), {0, 1})
        self.assertEqual(matches[0].coords, matches[1].coords)
        self.assertEqual(masks, {0: [True], 1: [True]})

    def test_equal_ious_select_distinct_anchors(self):
        matches, ious, masks = anchor.get_gt_max_iou(
            [0, 0, 10, 10, "cat"],
            [[0, 0, 10, 10], [0, 0, 10, 10]],
            anchor_labels=["anchor-a", "anchor-b"],
            positive_boxes=2,
            update_labels=False,
            return_ious=True,
            return_masks=True,
        )

        self.assertEqual(matches[0].label, ["anchor-a", "anchor-b"])
        self.assertEqual(ious[0], [1.0, 1.0])
        self.assertEqual(masks[0], [True, True])

    def test_matching_rejects_invalid_box_ids(self):
        with self.assertRaisesRegex(ValueError, "one box_id"):
            anchor.get_gt_max_iou(
                [[0, 0, 10, 10, "cat"], [20, 20, 30, 30, "cat"]],
                [[0, 0, 10, 10]],
                box_ids=["only-one"],
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            anchor.get_gt_max_iou(
                [[0, 0, 10, 10, "cat"], [20, 20, 30, 30, "cat"]],
                [[0, 0, 10, 10]],
                box_ids=["duplicate", "duplicate"],
            )


if __name__ == "__main__":
    unittest.main()
