import cv2
import numpy as np
import torch
from ultralytics import YOLO # type: ignore
import sys
import os
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from threading import Thread
import logging
import queue
import time

# -- FIXED IMPORTS --
# We assume we are running this script from the project root or src folder.
# This fixes the "Import src.config could not be resolved" error.
try:
    import config
    from diy_fusion import DIYFusionSystem
    from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    # Fallback if running from root directory without -m
    import src.config as config
    from src.diy_fusion import DIYFusionSystem
    from src.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pv.global_theme.allow_empty_mesh = True

import datetime

class RealTime3DViewer:
    def __init__(self, title="Real-time 3D Reconstruction"):
        self.plotter = BackgroundPlotter(title=title, auto_update=True)
        self.plotter.add_axes()
        self.actor_name = 'global_cloud'
        self._last_poly_data = pv.PolyData()
        
        # Initialize with a small dummy point to prevent PyVista warnings
        self.plotter.add_mesh(self._last_poly_data, name=self.actor_name)
        
        # Add a key press event for 's' to save the point cloud
        self.plotter.add_key_event('s', self.save_point_cloud)

    def update_and_render(self, points, colors=None):
        if points is None or len(points) == 0:
            return
            
        if colors is not None:
            if colors.dtype != np.uint8:
                colors = (colors * 255).astype(np.uint8)
        else:
            colors = np.full((points.shape[0], 3), [255, 255, 255], dtype=np.uint8)

        # Update the existing mesh
        self._last_poly_data = pv.PolyData(points)
        self._last_poly_data['colors'] = colors
        
        self.plotter.add_mesh(
            self._last_poly_data,
            name=self.actor_name,
            scalars='colors',
            rgb=True,
            point_size=3.0, # Smaller points look better for dense clouds
            render_points_as_spheres=True,
        )

    def save_point_cloud(self):
        """Saves the current point cloud to a PLY file with a timestamp."""
        if self._last_poly_data.n_points > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reconstruction_{timestamp}.ply"
            self._last_poly_data.save(filename)
            logging.info(f"Point cloud saved to {filename}")
        else:
            logging.warning("No point cloud to save.")

    def stop(self):
        self.plotter.close()

class VisionSystem:
    def __init__(self):
        self.yolo = None
        self.depth_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Vision system operating on device: {self.device}")
        self.cached_grid = None
        self.cached_shape = None

    def load_models(self):
        # Load YOLO
        if os.path.exists(config.YOLO_MODEL_PATH):
            logging.info(f"Loading YOLO from {config.YOLO_MODEL_PATH}")
            self.yolo = YOLO(config.YOLO_MODEL_PATH)
        else:
            logging.error(f"YOLO model missing at {config.YOLO_MODEL_PATH}")
            sys.exit(1)

        # Load Depth Anything V2
        self.depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        if os.path.exists(config.DEPTH_MODEL_PATH):
            logging.info(f"Loading Depth Model from {config.DEPTH_MODEL_PATH}")
            self.depth_model.load_state_dict(torch.load(config.DEPTH_MODEL_PATH, map_location='cpu'))
        else:
            logging.error(f"Depth model missing at {config.DEPTH_MODEL_PATH}")
            sys.exit(1)
        self.depth_model = self.depth_model.to(self.device).eval()

    def get_depth_map(self, frame):
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame)
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        return depth

    def get_3d_points(self, frame, depth_map, camera_matrix):
        """
        Converts 2D Image + Depth Map -> 3D Point Cloud (Local Coordinates)
        """
        height, width = depth_map.shape
        
        if self.cached_grid is None or self.cached_shape != (height, width):
            self.cached_shape = (height, width)
            u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
            self.cached_u = u_coords.flatten()
            self.cached_v = v_coords.flatten()
            self.cached_grid = True 
            
        # Filter invalid depth
        depth_flat = depth_map.flatten()
        # Invert depth (DepthAnything outputs relative inverse depth often, but check model specifics)
        # For DA-v2, raw output is metric-like but relative.
        # We assume standard output: Higher value = Closer? 
        # Actually DA-v2 output is often disparity. Let's invert carefully.
        z_valid = depth_flat > 0
        
        # Simple scaling for visualization
        z_3d = depth_flat[z_valid] * (config.DEPTH_SCALE / 50.0) 
        
        u_flat = self.cached_u[z_valid]
        v_flat = self.cached_v[z_valid]
        
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        x_3d = (u_flat - cx) * z_3d / fx
        y_3d = (v_flat - cy) * z_3d / fy # Invert Y if needed
        
        points_3d = np.stack([x_3d, z_3d, y_3d], axis=-1)
        
        # Get Colors
        color_flat = frame.reshape(-1, 3)
        colors = color_flat[z_valid]
        colors = colors[:, [2, 1, 0]] # BGR to RGB
        
        # Performance optimization: Don't return ALL points, just a subset
        if len(points_3d) > 5000:
            step = len(points_3d) // 5000
            return points_3d[::step], colors[::step]
        
        return points_3d, colors

# Shared queue
latest_frame_queue = queue.Queue(maxsize=1)

# --- ADD THIS IMPORT AT THE TOP ---
try:
    from diy_fusion import DIYFusionSystem
except ImportError:
    from src.diy_fusion import DIYFusionSystem

# --- REPLACE THE ENTIRE processing_thread FUNCTION ---
def processing_thread(vision_system, viewer, camera_matrix):
    """
    Updated processing thread that uses DIYFusionSystem to build a persistent world.
    """
    logging.info("Initializing DIY Fusion System...")
    # 1. Initialize the Fusion System (The "Memory")
    # voxel_size=0.05 means 5cm blocks. 
    fusion_system = DIYFusionSystem(voxel_size=0.05) 

    while True:
        try:
            frame = latest_frame_queue.get()
            if frame is None: break

            # 2. AI Inference
            depth_map = vision_system.get_depth_map(frame)
            
            # 3. Get Local 3D Points (The "Pin Art" for this specific frame)
            # We ignore 'detections' here as we just want the raw cloud
            points_3d, colors = vision_system.get_3d_points(frame, depth_map, camera_matrix)

            # 4. FUSE into Global World
            # This takes the new frame and "glues" it to the existing world
            if len(points_3d) > 0:
                global_pts, global_cols = fusion_system.process_frame(points_3d, colors, frame, camera_matrix)
                
                # 5. Render the Global World (not just the current frame)
                # We limit to 500,000 points to keep the viewer smooth
                if len(global_pts) > 500000:
                    # Simple random downsample if it gets too huge
                    indices = np.random.choice(len(global_pts), 500000, replace=False)
                    viewer.update_and_render(global_pts[indices], global_cols[indices])
                else:
                    viewer.update_and_render(global_pts, global_cols)

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in processing thread: {e}")

def main():
    logging.info("Starting System...")
    vision = VisionSystem()
    vision.load_models()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    viewer = RealTime3DViewer()

    # Setup Camera Matrix (Intrinsics)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fx = fy = max(W, H)
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Start Processing Thread
    proc_thread = Thread(target=processing_thread, args=(vision, viewer, K))
    proc_thread.daemon = True
    proc_thread.start()

    logging.info("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            cv2.imshow('Camera Feed (2D)', frame)

            # Non-blocking put
            if latest_frame_queue.full():
                try: latest_frame_queue.get_nowait()
                except queue.Empty: pass
            latest_frame_queue.put(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        latest_frame_queue.put(None)
        proc_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        viewer.stop()

if __name__ == "__main__":
    main()
