"""
Object-Centric Fusion System
Per-object 3D reconstruction with temporal fusion and tracking.
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

@dataclass
class TrackedObject:
    """Represents a tracked object in the 3D scene."""
    object_id: int
    label: str
    confidence: float
    
    # 3D representation
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    colors: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    
    # Tracking state
    last_seen_frame: int = 0
    total_frames_seen: int = 0
    bbox_history: List[np.ndarray] = field(default_factory=list)
    
    # Fusion parameters
    voxel_size: float = 1.0
    max_points: int = 5000
    
    def add_points(self, new_points: np.ndarray, new_colors: np.ndarray, frame_id: int):
        """
        Add new points to this object's 3D representation.
        Uses voxel filtering to prevent unbounded growth.
        """
        if len(new_points) == 0:
            return
            
        # Update tracking state
        self.last_seen_frame = frame_id
        self.total_frames_seen += 1
        
        # Combine with existing points
        if len(self.points) == 0:
            self.points = new_points.astype(np.float32)
            self.colors = new_colors.astype(np.uint8)
        else:
            self.points = np.vstack((self.points, new_points.astype(np.float32)))
            self.colors = np.vstack((self.colors, new_colors.astype(np.uint8)))
        
        # Apply voxel filtering to prevent memory explosion
        self._voxel_downsample()
    
    def _voxel_downsample(self):
        """Keep only one point per voxel cell (most recent overwrites)."""
        if len(self.points) <= self.max_points:
            return
            
        # Quantize to voxel grid
        quantized = (self.points / self.voxel_size).astype(np.int32)
        
        # Find unique voxels (keep last occurrence = most recent point)
        _, unique_indices = np.unique(quantized[::-1], axis=0, return_index=True)
        unique_indices = len(self.points) - 1 - unique_indices  # Reverse indices
        
        self.points = self.points[unique_indices]
        self.colors = self.colors[unique_indices]
    
    def get_centroid(self) -> Optional[np.ndarray]:
        """Get the 3D centroid of this object."""
        if len(self.points) == 0:
            return None
        return np.mean(self.points, axis=0)
    
    def get_bbox_3d(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get 3D axis-aligned bounding box (min, max)."""
        if len(self.points) == 0:
            return None
        return np.min(self.points, axis=0), np.max(self.points, axis=0)


class ObjectCentricFusion:
    """
    Manages per-object 3D reconstruction with temporal tracking.
    
    Key improvements over DIYFusionSystem:
    - Per-object point clouds (not global soup)
    - Object tracking by label + bbox overlap
    - Individual voxel filtering per object
    - Object lifecycle management (create, update, expire)
    """
    
    def __init__(self, voxel_size: float = 1.0, max_objects: int = 50, 
                 expire_after_frames: int = 30):
        """
        Args:
            voxel_size: Voxel resolution for downsampling
            max_objects: Maximum tracked objects
            expire_after_frames: Remove objects not seen for this many frames
        """
        self.voxel_size = voxel_size
        self.max_objects = max_objects
        self.expire_after_frames = expire_after_frames
        
        self.objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        self.frame_count = 0
        
        # Visual odometry for camera pose
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.last_keypoints = None
        self.last_descriptors = None
        self.current_pose = np.eye(4, dtype=np.float32)
        self.K = None  # Camera intrinsics
    
    def _estimate_camera_motion(self, frame: np.ndarray) -> bool:
        """Estimate camera motion using ORB features."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.last_keypoints is None:
            self.last_keypoints, self.last_descriptors = keypoints, descriptors
            return True
        
        if descriptors is None or len(keypoints) < 8:
            return False
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.last_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 8:
            return False
        
        # Extract matched points
        pts1 = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate Essential Matrix
        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, 
                                        prob=0.999, threshold=1.0)
        if E is None:
            return False
        
        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, self.K)
        
        # Update pose
        T_relative = np.eye(4, dtype=np.float32)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.reshape(3)
        self.current_pose = self.current_pose @ T_relative
        
        self.last_keypoints, self.last_descriptors = keypoints, descriptors
        return True
    
    def _compute_bbox_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def _find_matching_object(self, label: str, bbox: np.ndarray, 
                               iou_threshold: float = 0.3) -> Optional[int]:
        """Find existing object that matches this detection."""
        best_match_id = None
        best_iou = iou_threshold
        
        for obj_id, obj in self.objects.items():
            # Must be same class
            if obj.label != label:
                continue
            
            # Check bbox overlap with last known position
            if obj.bbox_history:
                iou = self._compute_bbox_iou(bbox, obj.bbox_history[-1])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id
        
        return best_match_id
    
    def _transform_points_to_world(self, points: np.ndarray) -> np.ndarray:
        """Transform local points to world coordinates using current pose."""
        R = self.current_pose[:3, :3]
        t = self.current_pose[:3, 3]
        return (R @ points.T).T + t
    
    def process_frame(self, frame: np.ndarray, 
                      detections: List[Tuple[np.ndarray, str, float, np.ndarray]],
                      points_per_detection: List[Tuple[np.ndarray, np.ndarray]],
                      camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a frame with detected objects.
        
        Args:
            frame: BGR image
            detections: List of (mask, label, confidence, bbox)
            points_per_detection: List of (points_3d, colors) matching detections
            camera_matrix: Camera intrinsics
            
        Returns:
            all_points: Combined points from all objects
            all_colors: Combined colors from all objects
        """
        self.frame_count += 1
        
        if self.K is None:
            self.K = camera_matrix
        
        # Update camera pose
        self._estimate_camera_motion(frame)
        
        # Process each detection
        for i, (mask, label, score, bbox) in enumerate(detections):
            if score < 0.5:
                continue
            
            pts, cols = points_per_detection[i] if i < len(points_per_detection) else (np.empty((0,3)), np.empty((0,3)))
            
            if len(pts) == 0:
                continue
            
            # Transform to world coordinates
            world_pts = self._transform_points_to_world(pts)
            
            # Find or create object
            match_id = self._find_matching_object(label, bbox)
            
            if match_id is not None:
                # Update existing object
                obj = self.objects[match_id]
                obj.add_points(world_pts, cols, self.frame_count)
                obj.bbox_history.append(bbox.copy())
                obj.confidence = max(obj.confidence, score)
            else:
                # Create new object
                if len(self.objects) < self.max_objects:
                    new_obj = TrackedObject(
                        object_id=self.next_object_id,
                        label=label,
                        confidence=score,
                        voxel_size=self.voxel_size
                    )
                    new_obj.add_points(world_pts, cols, self.frame_count)
                    new_obj.bbox_history.append(bbox.copy())
                    self.objects[self.next_object_id] = new_obj
                    self.next_object_id += 1
                    logging.debug(f"Created new object {new_obj.object_id}: {label}")
        
        # Expire old objects
        expired_ids = [
            obj_id for obj_id, obj in self.objects.items()
            if self.frame_count - obj.last_seen_frame > self.expire_after_frames
        ]
        for obj_id in expired_ids:
            logging.debug(f"Expiring object {obj_id}: {self.objects[obj_id].label}")
            del self.objects[obj_id]
        
        # Combine all object points for rendering
        return self.get_all_points()
    
    def get_all_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get combined point cloud from all tracked objects."""
        if not self.objects:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        
        all_pts = []
        all_cols = []
        
        for obj in self.objects.values():
            if len(obj.points) > 0:
                all_pts.append(obj.points)
                all_cols.append(obj.colors)
        
        if not all_pts:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        
        return np.vstack(all_pts), np.vstack(all_cols)
    
    def get_object_list(self) -> List[Dict]:
        """Get list of tracked objects with metadata."""
        return [
            {
                "id": obj.object_id,
                "label": obj.label,
                "confidence": obj.confidence,
                "point_count": len(obj.points),
                "frames_seen": obj.total_frames_seen,
                "last_seen": obj.last_seen_frame
            }
            for obj in self.objects.values()
        ]
    
    def clear(self):
        """Reset all tracked objects."""
        self.objects.clear()
        self.next_object_id = 0
        self.frame_count = 0
        self.current_pose = np.eye(4, dtype=np.float32)
        self.last_keypoints = None
        self.last_descriptors = None
