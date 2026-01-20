import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Setup path and mocks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.modules['depth_anything_v2'] = MagicMock()
sys.modules['depth_anything_v2.depth_anything_v2'] = MagicMock()
sys.modules['depth_anything_v2.depth_anything_v2.dpt'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()

from main import VisionSystem

class TestOptimizationCorrectness(unittest.TestCase):
    def setUp(self):
        self.vision = VisionSystem()
        self.H, self.W = 480, 640
        self.frame = np.random.randint(0, 255, (self.H, self.W, 3), dtype=np.uint8)
        self.depth_map = np.random.rand(self.H, self.W).astype(np.float32) * 10.0

        # Camera matrix
        fx = fy = 500
        cx, cy = self.W/2, self.H/2
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def test_masked_points_consistency(self):
        """
        Verify that using a bbox produces the same 3D points as full frame processing
        for the masked region.
        """
        # Create a mask in a specific region
        mask = np.zeros((self.H, self.W), dtype=bool)
        y1, x1, y2, x2 = 100, 200, 300, 400
        mask[y1:y2, x1:x2] = True

        # 1. Full frame processing (no bbox)
        points_full, colors_full = self.vision.get_3d_points_masked(self.frame, self.depth_map, mask, self.K)

        # 2. Optimized processing (with bbox covering the mask)
        bbox = [x1, y1, x2, y2]
        points_opt, colors_opt = self.vision.get_3d_points_masked(self.frame, self.depth_map, mask, self.K, bbox=bbox)

        # Check shapes
        self.assertEqual(points_full.shape, points_opt.shape)
        self.assertEqual(colors_full.shape, colors_opt.shape)

        # Check values
        np.testing.assert_allclose(points_full, points_opt, rtol=1e-5)
        np.testing.assert_allclose(colors_full, colors_opt, rtol=1e-5)

    def test_loose_bbox(self):
        """
        Verify that a bbox larger than the mask still works correctly.
        """
        mask = np.zeros((self.H, self.W), dtype=bool)
        y1, x1, y2, x2 = 100, 200, 300, 400
        mask[y1:y2, x1:x2] = True

        # Bbox slightly larger
        bbox = [x1-10, y1-10, x2+10, y2+10]

        points_full, colors_full = self.vision.get_3d_points_masked(self.frame, self.depth_map, mask, self.K)
        points_opt, colors_opt = self.vision.get_3d_points_masked(self.frame, self.depth_map, mask, self.K, bbox=bbox)

        np.testing.assert_allclose(points_full, points_opt, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
