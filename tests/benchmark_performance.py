import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unittest.mock import MagicMock
sys.modules['depth_anything_v2'] = MagicMock()
sys.modules['depth_anything_v2.depth_anything_v2'] = MagicMock()
sys.modules['depth_anything_v2.depth_anything_v2.dpt'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()

from main import VisionSystem

def benchmark():
    vision = VisionSystem()
    # No need to load models, get_3d_points_masked uses only numpy and args

    H, W = 1080, 1920
    frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth_map = np.random.rand(H, W).astype(np.float32) * 10.0

    # Create a small mask (ROI)
    mask = np.zeros((H, W), dtype=bool)
    # 200x200 object
    y_start, x_start = 400, 800
    h_obj, w_obj = 200, 200
    mask[y_start:y_start+h_obj, x_start:x_start+w_obj] = True

    # Camera matrix
    fx = fy = 1000
    cx, cy = W/2, H/2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        vision.get_3d_points_masked(frame, depth_map, mask, K)

    # Benchmark
    print("Running benchmark (Current Implementation)...")
    iterations = 50
    start_time = time.time()
    for _ in range(iterations):
        vision.get_3d_points_masked(frame, depth_map, mask, K)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"Average time per call: {avg_time*1000:.4f} ms")

    # Helper to test optimized version if available
    bbox = [x_start, y_start, x_start+w_obj, y_start+h_obj]

    # Benchmark Optimized
    print("Running benchmark (Optimized with bbox)...")
    start_time = time.time()
    for _ in range(iterations):
        vision.get_3d_points_masked(frame, depth_map, mask, K, bbox=bbox)
    end_time = time.time()

    avg_time_opt = (end_time - start_time) / iterations
    print(f"Average time per call (Optimized): {avg_time_opt*1000:.4f} ms")

if __name__ == "__main__":
    benchmark()
