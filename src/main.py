import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
import requests
import json
import sys
import os
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import BackgroundPlotter
from threading import Thread, Lock
import logging
import time
import queue
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import src.config as config

class RealTime3DViewer:
    """
    A thread-safe viewer that uses pyvista.BackgroundPlotter to display a
    point cloud that can be updated from an external thread.
    """
    def __init__(self, title="Real-time 3D Point Cloud Viewer"):
        self.title = title
        # BackgroundPlotter handles its own window, thread, and updates.
        self.plotter = BackgroundPlotter(title=self.title, auto_update=True)
        self.plotter.add_axes()
        # Name for our actor to reference it for updates
        self.actor_name = 'point_cloud'
        # Add an initial empty point cloud
        self.plotter.add_mesh(pv.PolyData(), name=self.actor_name)

    def update_and_render(self, points, colors=None):
        """
        Updates the point cloud mesh in the plotter. This method is thread-safe
        as BackgroundPlotter is designed for this purpose.
        """
        if points is None or len(points) == 0:
            return
            
        if colors is not None:
            if colors.dtype != np.uint8:
                colors = (colors * 255).astype(np.uint8)
        else:
            # Default to white if no colors are provided
            colors = np.full((points.shape[0], 3), [255, 255, 255], dtype=np.uint8)

        poly_data = pv.PolyData(points)
        poly_data['colors'] = colors
        
        # Re-add the mesh with the same name to update it
        self.plotter.add_mesh(
            poly_data,
            name=self.actor_name,
            scalars='colors',
            rgb=True,
            point_size=5.0,
            render_points_as_spheres=True,
        )

    def stop(self):
        """ Closes the plotter window. """
        self.plotter.close()

    def is_active(self):
        """ Checks if the plotter is running. """
        # BackgroundPlotter manages its own window state.
        # We assume it's active until stop() is called.
        return self.plotter is not None

class VisionSystem:
    def __init__(self):
        self.yolo = None
        self.depth_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Vision system operating on device: {self.device}")
        self.cached_grid = None
        self.cached_shape = None

    def load_models(self):
        # Load YOLO model
        if os.path.exists(config.YOLO_MODEL_PATH):
            logging.info(f"Loading YOLO model from {config.YOLO_MODEL_PATH}")
            self.yolo = YOLO(config.YOLO_MODEL_PATH)
        else:
            logging.error(f"YOLO model not found at {config.YOLO_MODEL_PATH}. Please download the model.")
            sys.exit(1)

        # Load Depth Anything V2 model
        from src.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
        self.depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        if os.path.exists(config.DEPTH_MODEL_PATH):
            logging.info(f"Loading Depth Anything V2 model from {config.DEPTH_MODEL_PATH}")
            self.depth_model.load_state_dict(torch.load(config.DEPTH_MODEL_PATH, map_location='cpu'))
        else:
            logging.error(f"Depth Anything V2 model not found at {config.DEPTH_MODEL_PATH}. Please run download_depth_anything_v2.py to download the model weights.")
            sys.exit(1)
        self.depth_model = self.depth_model.to(self.device).eval()

    def get_depth_map(self, frame):
        with torch.no_grad():
            depth = self.depth_model.infer_image(frame)
            if depth.shape[:2] != frame.shape[:2]:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return depth

    def process_frame(self, frame, depth_map, camera_matrix):
        try:
            if frame is None or frame.size == 0:
                logging.warning("Input frame is empty.")
                return [], [], []
            if depth_map is None or depth_map.size == 0:
                logging.warning("Depth map is empty.")
                return [], [], []

            H, W = frame.shape[:2]
            results = self.yolo(frame, verbose=False)
            detections = []
            for det in results[0].boxes:
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                conf = float(det.conf[0])
                cls_id = int(det.cls[0])
                if conf < 0.5:
                    continue
                cx_obj, cy_obj = int((x1 + x2) / 2), int((y1 + y2) / 2)
                z_norm = depth_map[cy_obj, cx_obj]
                z_metric = 1.0 / (z_norm + 1e-5)
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                x_3d = (cx_obj - cx) * z_metric / fx
                y_3d = (cy_obj - cy) * z_metric / fy
                detections.append({
                    'class_id': cls_id,
                    'conf': conf,
                    'bbox_2d': (x1, y1, x2, y2),
                    'position_3d': (x_3d, y_3d, z_metric),
                    'bbox_2d_area': (x2 - x1) * (y2 - y1)
                })
            points_3d, colors = self.create_3d_points_from_depth(frame, depth_map, camera_matrix)
            return detections, points_3d, colors
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return [], [], []

    def create_3d_points_from_depth(self, frame, depth_map, camera_matrix):
        height, width = depth_map.shape
        
        # Only compute meshgrid if shape changes
        if self.cached_grid is None or self.cached_shape != (height, width):
            self.cached_shape = (height, width)
            u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
            self.cached_u = u_coords.flatten()
            self.cached_v = v_coords.flatten()
            self.cached_grid = True # Mark cache as built
            
        # Use cached values
        depth_valid = 1.0 / (depth_map.flatten() + 0.001)
        valid_mask = (depth_valid < 1000) & (depth_valid > 0.1)
        
        u_flat = self.cached_u[valid_mask]
        v_flat = self.cached_v[valid_mask]
        
        z_3d = depth_valid[valid_mask] * config.DEPTH_SCALE
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        x_3d = (u_flat - cx) * z_3d / fx
        y_3d = -(v_flat - cy) * z_3d / fy
        points_3d = np.stack([x_3d, z_3d, y_3d], axis=-1)
        color_flat = frame.reshape(-1, 3)
        colors = color_flat[valid_mask]
        colors = colors[:, [2, 1, 0]]
        return points_3d, colors

# Shared queue for the latest frame
latest_frame_queue = queue.Queue(maxsize=1)

def processing_thread(vision_system, viewer, camera_matrix):
    """
    This thread function grabs frames from the queue, performs heavy AI processing,
    and updates the 3D viewer.
    """
    while True:
        try:
            # Get the latest frame, waiting if the queue is empty
            frame = latest_frame_queue.get()
            if frame is None:  # Sentinel value to stop the thread
                break

            # --- Heavy Lifting Here ---
            depth_map = vision_system.get_depth_map(frame)
            detections, points_3d, colors = vision_system.process_frame(frame, depth_map, camera_matrix)

            # Update 3D Viewer
            if len(points_3d) > 0:
                # Downsample if necessary to maintain performance
                if len(points_3d) > 5000:
                    step = len(points_3d) // 5000
                    points_3d = points_3d[::step]
                    colors = colors[::step]
                viewer.update_and_render(points_3d, colors)
        except queue.Empty:
            continue # Loop again if the queue was empty
        except Exception as e:
            logging.error(f"Error in processing thread: {e}")


def main():
    # Initialize the vision system
    logging.info("Initializing vision system...")
    vision_system = VisionSystem()
    vision_system.load_models()

    # Initialize the 3D viewer
    logging.info("Initializing 3D viewer...")
    viewer = RealTime3DViewer()
    # No longer need to call viewer.start() as BackgroundPlotter is now used

    # Initialize camera
    logging.info(f"Initializing camera at index {config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    # Camera intrinsic parameters
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fx = fy = max(width, height)
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    logging.info("Starting processing thread...")
    process_thread = Thread(target=processing_thread, args=(vision_system, viewer, camera_matrix))
    process_thread.daemon = True
    process_thread.start()

    logging.info("Press 'q' in the camera window to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. SHOW 2D IMMEDIATELY (Real-time!)
            cv2.imshow('Camera Feed', frame)

            # 2. Send frame to AI thread (Non-blocking)
            # If queue is full, it means the AI is still working on the previous frame.
            # We empty the queue and put the new frame, ensuring the AI always gets the latest data.
            if latest_frame_queue.full():
                try:
                    # Non-blocking get to remove the old frame
                    latest_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                 # Non-blocking put to add the new frame
                latest_frame_queue.put_nowait(frame)
            except queue.Full:
                pass


            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        logging.info("Cleaning up...")
        # Stop the processing thread
        latest_frame_queue.put(None)
        process_thread.join()

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        viewer.stop()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()