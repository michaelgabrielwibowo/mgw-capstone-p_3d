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

# Direct imports - run this script from the src directory or use 'python -m src.main' from root
import config
from diy_fusion import DIYFusionSystem
from object_fusion import ObjectCentricFusion  # Legacy fallback
from delta_fusion import DeltaFusionSystem  # New delta-compression system
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from sam_segmenter import SAMSegmenter
from ui_overlay import UIOverlay
from export_utils import export_ply

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pv.global_theme.allow_empty_mesh = True

class RealTime3DViewer:
    def __init__(self, title="Real-time 3D Reconstruction"):
        self.plotter = BackgroundPlotter(title=title, auto_update=True)
        self.plotter.add_axes()
        self.actor_name = 'global_cloud'
        
        # Initialize with a small dummy point to prevent PyVista warnings
        self.plotter.add_mesh(pv.PolyData(np.array([[0.,0.,0.]])), name=self.actor_name)

    def update_and_render(self, points, colors=None):
        if points is None or len(points) == 0:
            return
            
        if colors is not None:
            if colors.dtype != np.uint8:
                colors = (colors * 255).astype(np.uint8)
        else:
            colors = np.full((points.shape[0], 3), [255, 255, 255], dtype=np.uint8)

        # Update the existing mesh
        poly_data = pv.PolyData(points)
        poly_data['colors'] = colors
        
        self.plotter.add_mesh(
            poly_data,
            name=self.actor_name,
            scalars='colors',
            rgb=True,
            point_size=3.0, # Smaller points look better for dense clouds
            render_points_as_spheres=True,
        )

    def stop(self):
        self.plotter.close()

class VisionSystem:
    def __init__(self):
        self.yolo = None
        self.depth_model = None
        self.sam = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Vision system operating on device: {self.device}")
        self.cached_grid = None
        self.cached_shape = None
        self.cached_u = None
        self.cached_v = None

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

        # Load SAM (Segment Anything Model)
        if os.path.exists(config.SAM_MODEL_PATH):
            logging.info(f"Loading SAM from {config.SAM_MODEL_PATH}")
            self.sam = SAMSegmenter(model_type=config.SAM_MODEL_TYPE, device=str(self.device))
            self.sam.load_model(config.SAM_MODEL_PATH)
        else:
            logging.warning(f"SAM model missing at {config.SAM_MODEL_PATH} - running without segmentation")

    def detect_and_segment(self, frame):
        """
        Detect objects with YOLO and segment with SAM.
        
        Returns:
            List of tuples: (mask, label, confidence, bbox) for each detected object
        """
        # Convert BGR to RGB for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = self.yolo(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)
        scores = results.boxes.conf.cpu().numpy()
        names = results.names  # Class name mapping
        
        if len(boxes) == 0:
            return []
        
        # Get SAM masks for each detection
        if self.sam is not None:
            masks = self.sam.get_masks_for_detections(rgb_frame, boxes)
        else:
            # Fallback: use bounding box as rectangular mask
            masks = []
            h, w = frame.shape[:2]
            for box in boxes:
                mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = True
                masks.append(mask)
        
        detections = []
        for i in range(len(boxes)):
            label_name = names[labels[i]]
            detections.append((masks[i], label_name, scores[i], boxes[i]))
            logging.debug(f"Detected: {label_name} ({scores[i]:.2f})")
        
        return detections

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

    def get_3d_points_masked(self, frame, depth_map, mask, camera_matrix):
        """
        Converts 2D Image + Depth Map -> 3D Point Cloud (ONLY for masked region)
        
        Args:
            frame: BGR image
            depth_map: Depth values for each pixel
            mask: Binary mask (H, W) where True = include pixel
            camera_matrix: Camera intrinsics
            
        Returns:
            points_3d: (N, 3) array of 3D points
            colors: (N, 3) array of RGB colors
        """
        height, width = depth_map.shape
        
        # Performance optimization: Cache coordinate grids to avoid re-allocation
        if self.cached_grid is None or self.cached_shape != (height, width):
            self.cached_shape = (height, width)
            u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
            self.cached_u = u_coords.flatten()
            self.cached_v = v_coords.flatten()
            self.cached_grid = True

        u_flat = self.cached_u
        v_flat = self.cached_v
        
        # Flatten mask and depth
        mask_flat = mask.flatten()
        depth_flat = depth_map.flatten()
        
        # Combined validity: inside mask AND valid depth
        valid = mask_flat & (depth_flat > 0)
        
        if not np.any(valid):
            return np.empty((0, 3)), np.empty((0, 3))
        
        # Get valid coordinates and depth
        z_3d = depth_flat[valid] * (config.DEPTH_SCALE / 50.0)
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]
        
        # Camera intrinsics
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # Project to 3D
        x_3d = (u_valid - cx) * z_3d / fx
        y_3d = (v_valid - cy) * z_3d / fy
        
        points_3d = np.stack([x_3d, z_3d, y_3d], axis=-1)
        
        # Get Colors
        color_flat = frame.reshape(-1, 3)
        colors = color_flat[valid]
        colors = colors[:, [2, 1, 0]]  # BGR to RGB
        
        # Performance optimization: subsample if too many points
        max_points = 2000  # Per object
        if len(points_3d) > max_points:
            step = len(points_3d) // max_points
            return points_3d[::step], colors[::step]
        
        return points_3d, colors

# Shared queue and fusion system
latest_frame_queue = queue.Queue(maxsize=1)
shared_fusion_system = None  # Will be set in processing thread

def processing_thread(vision_system, viewer, camera_matrix):
    global shared_fusion_system
    
    # Initialize the Delta Fusion System (video-compression-style updates)
    logging.info("Initializing Delta Fusion System (Performance Mode)...")
    fusion_system = DeltaFusionSystem(
        voxel_size=config.VOXEL_SIZE,
        max_objects=config.MAX_OBJECTS,
        expire_after_frames=config.EXPIRE_AFTER_FRAMES,
        keyframe_interval=config.KEYFRAME_INTERVAL,
        stale_voxel_age=config.STALE_VOXEL_AGE
    )
    shared_fusion_system = fusion_system  # Make accessible to main thread
    
    frame_count = 0
    start_time = time.time()
    
    # Performance: Skip SAM on some frames (heavy operation)
    SAM_SKIP_FRAMES = config.SAM_SKIP_FRAMES
    RENDER_SKIP_FRAMES = config.RENDER_SKIP_FRAMES
    cached_detections = []
    cached_depth_map = None

    while True:
        try:
            frame = latest_frame_queue.get()
            if frame is None: break
            
            frame_count += 1
            
            # Performance optimization: Only run SAM every N frames
            run_full_detection = (frame_count % SAM_SKIP_FRAMES == 0) or len(cached_detections) == 0
            
            if run_full_detection:
                # 1. Detect and segment objects (heavy - SAM)
                detections = vision_system.detect_and_segment(frame)
                cached_detections = detections
                
                # 2. Get depth map
                depth_map = vision_system.get_depth_map(frame)
                cached_depth_map = depth_map
            else:
                # Use cached detections, but update depth map (it's fast)
                detections = cached_detections
                depth_map = vision_system.get_depth_map(frame)
            
            # 3. Get 3D points for each detected object
            points_per_detection = []
            for mask, label, score, bbox in detections:
                if score < 0.5:
                    points_per_detection.append((np.empty((0,3)), np.empty((0,3))))
                    continue
                pts, cols = vision_system.get_3d_points_masked(frame, depth_map, mask, camera_matrix)
                points_per_detection.append((pts, cols))
            
            # 4. Process through delta fusion
            global_pts, global_cols = fusion_system.process_frame(
                frame, detections, points_per_detection, camera_matrix
            )
            
            # 5. Render 3D view (skip some frames for smoother UI)
            if len(global_pts) > 0 and frame_count % RENDER_SKIP_FRAMES == 0:
                viewer.update_and_render(global_pts, global_cols)
            
            # Log performance
            if frame_count % 30 == 0:  # Less frequent logging
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                stats = fusion_system.get_stats()
                logging.info(f"FPS: {fps:.2f} | Objects: {stats['object_count']} | Voxels: {stats['total_voxels']}")

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in processing: {e}")
            import traceback
            traceback.print_exc()

def main():
    logging.info("Starting System...")
    vision = VisionSystem()
    vision.load_models()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    viewer = RealTime3DViewer()
    ui = UIOverlay()

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

    # UI state
    show_masks = True
    show_labels = True
    show_bboxes = True
    show_help = True
    current_detections = []

    # Mouse callback for add-object mode
    def mouse_callback(event, x, y, flags, param):
        ui.handle_mouse_event(event, x, y, flags)
    
    cv2.namedWindow('Object Detection (2D)')
    cv2.setMouseCallback('Object Detection (2D)', mouse_callback)

    logging.info("Press 'H' for help, 'Q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Run detection in main thread for UI responsiveness
            # (Note: This runs every frame for UI overlay)
            current_detections = vision.detect_and_segment(frame)
            
            # Render overlays
            display_frame = ui.render_detections(
                frame, 
                current_detections,
                show_masks=show_masks,
                show_labels=show_labels,
                show_bboxes=show_bboxes
            )
            
            # Draw help controls if enabled
            if show_help:
                display_frame = ui.draw_controls_help(display_frame)
            
            # Draw add-object box if in that mode
            if ui.mode == "add_object":
                display_frame = ui.draw_add_object_box(display_frame)
            
            cv2.imshow('Object Detection (2D)', display_frame)

            # Non-blocking put to processing thread
            if latest_frame_queue.full():
                try: latest_frame_queue.get_nowait()
                except queue.Empty: pass
            latest_frame_queue.put(frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('m') or key == ord('M'):
                show_masks = not show_masks
                logging.info(f"Masks: {'ON' if show_masks else 'OFF'}")
            elif key == ord('l') or key == ord('L'):
                show_labels = not show_labels
                logging.info(f"Labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord('b') or key == ord('B'):
                show_bboxes = not show_bboxes
                logging.info(f"Bboxes: {'ON' if show_bboxes else 'OFF'}")
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help
            elif key == ord('a') or key == ord('A'):
                # Accept selected object
                if ui.selected_object_idx >= 0:
                    ui.accept_object(ui.selected_object_idx)
                    logging.info(f"Accepted object {ui.selected_object_idx}")
            elif key == ord('r') or key == ord('R'):
                # Reject selected object
                if ui.selected_object_idx >= 0:
                    ui.reject_object(ui.selected_object_idx)
                    logging.info(f"Rejected object {ui.selected_object_idx}")
            elif key == 9:  # Tab key
                ui.select_next_object(len(current_detections))
                logging.info(f"Selected object: {ui.selected_object_idx}")
            elif key == ord('e') or key == ord('E'):
                # Export point cloud
                if shared_fusion_system is not None:
                    pts, cols = shared_fusion_system.get_all_points()
                    if len(pts) > 0:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"export_{timestamp}.ply"
                        try:
                            export_ply(filename, pts, cols)
                            logging.info(f"✓ Exported {len(pts)} points to {filename}")
                        except Exception as e:
                            logging.error(f"Export failed: {e}")
                    else:
                        logging.warning("No points to export")
                else:
                    logging.warning("Fusion system not ready")
            elif key == ord(' '):  # Space
                if ui.mode == "view":
                    ui.toggle_mode("add_object")
                    logging.info("Add object mode: Draw a box around the object")
                else:
                    ui.toggle_mode("view")
                    logging.info("View mode")
            
            # Check for user-drawn box in add_object mode
            drawn_box = ui.get_drawn_box()
            if drawn_box:
                logging.info(f"User drew box: {drawn_box}")
                # Get SAM mask for user-drawn box
                if vision.sam is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    vision.sam.set_image(rgb_frame)
                    user_mask = vision.sam.get_mask_from_bbox(drawn_box)
                    # Add as new detection
                    current_detections.append((user_mask, "user_object", 1.0, np.array(drawn_box)))
                    logging.info("Added user-defined object")
                ui.toggle_mode("view")

    finally:
        latest_frame_queue.put(None)
        proc_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        viewer.stop()

if __name__ == "__main__":
    main()
