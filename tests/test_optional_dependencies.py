import unittest
from unittest.mock import patch

import numpy as np

from pybx import sample, vis


class OptionalDependenciesTestCase(unittest.TestCase):
    def test_opencv_error_explains_how_to_install_extra(self):
        with patch.object(
            sample.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named 'cv2'"),
        ):
            with self.assertRaisesRegex(ImportError, r"pybx\[opencv\]"):
                sample._get_resized(np.zeros((2, 2, 3)), (1, 1))

    def test_matplotlib_error_explains_how_to_install_extra(self):
        with patch.object(
            vis.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named 'matplotlib'"),
        ):
            with self.assertRaisesRegex(ImportError, r"pybx\[viz\]"):
                vis._require_matplotlib()


if __name__ == "__main__":
    unittest.main()
