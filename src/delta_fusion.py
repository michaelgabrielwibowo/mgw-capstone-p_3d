"""
Delta Fusion System - Video-Compression-Style 3D Point Cloud Updates

This module implements an incremental update system for 3D point clouds,
inspired by video compression techniques (I-frames and P-frames).

Key Concepts:
- Keyframe (I-Frame): Full point cloud snapshot, replaces all voxels in region
- Delta Frame (P-Frame): Only update changed/new voxels
- Spatial Hash Grid: O(1) lookup for voxel operations
- Stale Voxel Cleanup: Remove old points not seen recently
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum
import logging
import config


class BlendMode(Enum):
    """How to blend new points with existing voxel data."""
    REPLACE = "replace"      # New points overwrite old
    AVERAGE = "average"      # Weighted average of positions/colors
    NEWEST = "newest"        # Keep only most recent observation
    ACCUMULATE = "accumulate"  # Keep all (old behavior fallback)


@dataclass
class VoxelCell:
    """
    Represents a single voxel cell in the spatial grid.
    
    Each cell stores a single representative point (centroid) and color,
    plus metadata for temporal tracking.
    """
    key: Tuple[int, int, int]
    position: np.ndarray        # Representative 3D position (centroid)
    color: np.ndarray           # Representative RGB color
    last_update_frame: int      # When was this cell last updated
    is_keyframe: bool           # Was this from a keyframe update?
    observation_count: int = 1  # How many times we've seen this voxel
    confidence: float = 1.0     # Accumulated confidence


class SpatialHashGrid:
    """
    Spatial hash grid for O(1) voxel lookups and updates.
    
    This replaces linear array storage with a dictionary keyed by
    quantized 3D coordinates, enabling fast spatial queries.
    """
    
    def __init__(self, voxel_size: float = 1.0):
        """
        Args:
            voxel_size: Size of each voxel cell in world units
        """
        self.voxel_size = voxel_size
        self.grid: Dict[Tuple[int, int, int], VoxelCell] = {}
    
    def hash_point(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert a 3D point to its voxel key."""
        quantized = np.floor(point / self.voxel_size).astype(np.int32)
        return (int(quantized[0]), int(quantized[1]), int(quantized[2]))
    
    def get_cell(self, key: Tuple[int, int, int]) -> Optional[VoxelCell]:
        """Get the cell at the given voxel key, or None if empty."""
        return self.grid.get(key)
    
    def update_cell(self, key: Tuple[int, int, int], position: np.ndarray, 
                    color: np.ndarray, frame_id: int, is_keyframe: bool = False,
                    blend_mode: BlendMode = BlendMode.NEWEST) -> None:
        """
        Update or create a voxel cell.
        
        Args:
            key: Voxel grid coordinates
            position: 3D position of the new point
            color: RGB color of the new point
            frame_id: Current frame number
            is_keyframe: Whether this is a keyframe update
            blend_mode: How to handle existing data
        """
        existing = self.grid.get(key)
        
        if existing is None:
            # Create new cell
            self.grid[key] = VoxelCell(
                key=key,
                position=position.copy().astype(np.float32),
                color=color.copy().astype(np.uint8),
                last_update_frame=frame_id,
                is_keyframe=is_keyframe,
                observation_count=1
            )
        else:
            # Update existing cell based on blend mode
            if blend_mode == BlendMode.REPLACE or is_keyframe:
                existing.position = position.copy().astype(np.float32)
                existing.color = color.copy().astype(np.uint8)
            elif blend_mode == BlendMode.AVERAGE:
                # Weighted average (favor newer observations for temporal smoothing)
                weight = getattr(config, 'TEMPORAL_SMOOTHING_WEIGHT', 0.3)
                existing.position = (
                    (1 - weight) * existing.position + weight * position
                ).astype(np.float32)
                existing.color = (
                    (1 - weight) * existing.color.astype(np.float32) + 
                    weight * color.astype(np.float32)
                ).astype(np.uint8)
            elif blend_mode == BlendMode.NEWEST:
                # Just replace if this is a newer observation
                existing.position = position.copy().astype(np.float32)
                existing.color = color.copy().astype(np.uint8)
            # ACCUMULATE mode would need multiple points per cell (not implemented here)
            
            existing.last_update_frame = frame_id
            existing.is_keyframe = is_keyframe
            existing.observation_count += 1

    def update_cells(self, points: np.ndarray, colors: np.ndarray,
                     frame_id: int, is_keyframe: bool = False,
                     blend_mode: BlendMode = BlendMode.NEWEST) -> None:
        """
        Vectorized update of voxel cells.

        Args:
            points: (N, 3) array of new 3D points
            colors: (N, 3) array of RGB colors
            frame_id: Current frame number
            is_keyframe: Whether this is a keyframe update
            blend_mode: How to handle existing data
        """
        if len(points) == 0:
            return

        # Vectorized hashing: (N, 3) int array
        keys_arr = np.floor(points / self.voxel_size).astype(np.int32)

        # Convert to tuples for dictionary keys
        keys_tuples = [tuple(k) for k in keys_arr.tolist()]

        # Calculate counts of how many points land in each voxel
        # This preserves the observation_count behavior
        counts = Counter(keys_tuples)

        # For REPLACE/NEWEST: we want the last point for each unique key.
        # dict(zip(keys, indices)) naturally keeps the last index for each unique key.
        unique_indices_map = {k: i for i, k in enumerate(keys_tuples)}

        # Pre-allocate numpy arrays to avoid repeated casting inside loop if possible,
        # but we are storing into objects, so we need to copy anyway.

        for key, idx in unique_indices_map.items():
            pos = points[idx]
            col = colors[idx]
            count_inc = counts[key]

            existing = self.grid.get(key)

            if existing is None:
                # Create new cell
                self.grid[key] = VoxelCell(
                    key=key,
                    position=pos.astype(np.float32),
                    color=col.astype(np.uint8),
                    last_update_frame=frame_id,
                    is_keyframe=is_keyframe,
                    observation_count=count_inc
                )
            else:
                # Update existing
                if blend_mode == BlendMode.REPLACE or is_keyframe:
                    existing.position[:] = pos
                    existing.color[:] = col
                elif blend_mode == BlendMode.NEWEST:
                     existing.position[:] = pos
                     existing.color[:] = col
                elif blend_mode == BlendMode.AVERAGE:
                    # Partial support for average: fallback to simple average with the NEW point
                    # (Note: this ignores intermediate points if multiple mapped to same voxel in this batch)
                    weight = getattr(config, 'TEMPORAL_SMOOTHING_WEIGHT', 0.3)
                    existing.position = (
                        (1 - weight) * existing.position + weight * pos
                    ).astype(np.float32)
                    existing.color = (
                        (1 - weight) * existing.color.astype(np.float32) +
                        weight * col.astype(np.float32)
                    ).astype(np.uint8)

                existing.last_update_frame = frame_id
                existing.is_keyframe = is_keyframe
                existing.observation_count += count_inc
    
    def remove_cell(self, key: Tuple[int, int, int]) -> bool:
        """Remove a cell from the grid. Returns True if cell existed."""
        if key in self.grid:
            del self.grid[key]
            return True
        return False
    
    def remove_stale_cells(self, current_frame: int, max_age: int) -> int:
        """
        Remove cells that haven't been updated recently.
        
        Args:
            current_frame: Current frame number
            max_age: Maximum frames since last update before removal
            
        Returns:
            Number of cells removed
        """
        stale_keys = [
            key for key, cell in self.grid.items()
            if current_frame - cell.last_update_frame > max_age
        ]
        for key in stale_keys:
            del self.grid[key]
        return len(stale_keys)
    
    def get_cells_in_bbox(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get all cell keys within a 3D bounding box."""
        min_key = self.hash_point(bbox_min)
        max_key = self.hash_point(bbox_max)
        
        matching_keys = []
        for key in self.grid.keys():
            if (min_key[0] <= key[0] <= max_key[0] and
                min_key[1] <= key[1] <= max_key[1] and
                min_key[2] <= key[2] <= max_key[2]):
                matching_keys.append(key)
        return matching_keys
    
    def to_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the grid to point cloud arrays.
        
        Returns:
            points: (N, 3) array of positions
            colors: (N, 3) array of RGB colors
        """
        if not self.grid:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        
        points = np.array([cell.position for cell in self.grid.values()], dtype=np.float32)
        colors = np.array([cell.color for cell in self.grid.values()], dtype=np.uint8)
        return points, colors
    
    def __len__(self) -> int:
        return len(self.grid)


@dataclass
class DeltaFusionObject:
    """
    Per-object 3D representation using delta-compression updates.
    
    Instead of accumulating all points, this class:
    1. Maintains a spatial hash grid of voxels
    2. Applies delta updates (only change what's new)
    3. Periodically applies keyframes (full refresh)
    4. Cleans up stale voxels automatically
    """
    object_id: int
    label: str
    confidence: float
    
    # Spatial representation
    spatial_grid: SpatialHashGrid = field(default_factory=lambda: SpatialHashGrid(voxel_size=1.0))
    
    # Keyframe settings
    keyframe_interval: int = 30  # Full refresh every N frames
    last_keyframe_frame: int = 0
    
    # Tracking state
    last_seen_frame: int = 0
    total_frames_seen: int = 0
    bbox_history: List[np.ndarray] = field(default_factory=list)
    
    # 3D bounding box tracking (for keyframe region clearing)
    bbox_3d_min: Optional[np.ndarray] = None
    bbox_3d_max: Optional[np.ndarray] = None
    
    def update_points(self, new_points: np.ndarray, new_colors: np.ndarray,
                      frame_id: int, force_keyframe: bool = False) -> None:
        """
        Apply delta or keyframe update to this object's point cloud.
        
        Args:
            new_points: (N, 3) array of new 3D points
            new_colors: (N, 3) array of RGB colors
            frame_id: Current frame number
            force_keyframe: Force a keyframe update regardless of interval
        """
        if len(new_points) == 0:
            return
        
        # Update tracking state
        self.last_seen_frame = frame_id
        self.total_frames_seen += 1
        
        # Update 3D bounding box
        pts_min = np.min(new_points, axis=0)
        pts_max = np.max(new_points, axis=0)
        if self.bbox_3d_min is None:
            self.bbox_3d_min = pts_min
            self.bbox_3d_max = pts_max
        else:
            # Expand bbox if needed (with some smoothing)
            self.bbox_3d_min = np.minimum(self.bbox_3d_min * 0.9 + pts_min * 0.1, pts_min)
            self.bbox_3d_max = np.maximum(self.bbox_3d_max * 0.9 + pts_max * 0.1, pts_max)
        
        # Determine if this is a keyframe (adaptive strategy)
        frames_since_keyframe = frame_id - self.last_keyframe_frame
        time_based_keyframe = frames_since_keyframe >= self.keyframe_interval
        
        # Motion-based keyframe: trigger if object moved significantly
        motion_based_keyframe = False
        if len(new_points) > 0 and self.bbox_3d_min is not None:
            old_centroid = (self.bbox_3d_min + self.bbox_3d_max) / 2
            new_centroid = np.mean(new_points, axis=0)
            movement = np.linalg.norm(new_centroid - old_centroid)
            # Trigger keyframe if moved more than 10 voxels
            motion_threshold = self.spatial_grid.voxel_size * 10
            motion_based_keyframe = movement > motion_threshold
            if motion_based_keyframe:
                logging.debug(f"Object {self.object_id} ({self.label}): Motion keyframe (moved {movement:.1f} units)")
        
        is_keyframe = force_keyframe or time_based_keyframe or motion_based_keyframe
        
        if is_keyframe:
            self._apply_keyframe(new_points, new_colors, frame_id)
            self.last_keyframe_frame = frame_id
            keyframe_reason = "forced" if force_keyframe else ("motion" if motion_based_keyframe else "time")
            logging.debug(f"Object {self.object_id} ({self.label}): Keyframe applied ({keyframe_reason})")
        else:
            self._apply_delta(new_points, new_colors, frame_id)
    
    def _apply_keyframe(self, points: np.ndarray, colors: np.ndarray, frame_id: int) -> None:
        """
        Full replacement of points within the object's 3D bounding box.
        
        This clears all existing voxels in the object's region and replaces
        them with the new observations.
        """
        # Clear cells within the object's bounding box
        if self.bbox_3d_min is not None and self.bbox_3d_max is not None:
            # Add some padding to the bbox
            padding = self.spatial_grid.voxel_size * 2
            bbox_min_padded = self.bbox_3d_min - padding
            bbox_max_padded = self.bbox_3d_max + padding
            
            cells_to_remove = self.spatial_grid.get_cells_in_bbox(bbox_min_padded, bbox_max_padded)
            for key in cells_to_remove:
                self.spatial_grid.remove_cell(key)
        
        # Add all new points (Vectorized)
        self.spatial_grid.update_cells(
            points, colors, frame_id,
            is_keyframe=True, blend_mode=BlendMode.REPLACE
        )
    
    def _apply_delta(self, points: np.ndarray, colors: np.ndarray, frame_id: int) -> None:
        """
        Incremental update - only modify cells with new observations.
        
        For each new point:
        - If voxel is empty: Create new cell
        - If voxel exists: Update with blending (favor newer data)
        """
        # Use AVERAGE blend mode for temporal smoothing (reduces jitter)
        for i in range(len(points)):
            key = self.spatial_grid.hash_point(points[i])
            self.spatial_grid.update_cell(
                key, points[i], colors[i], frame_id,
                is_keyframe=False, blend_mode=BlendMode.AVERAGE
            )
    
    def cleanup_stale_voxels(self, current_frame: int, max_age: int = 90) -> int:
        """Remove voxels not updated recently."""
        return self.spatial_grid.remove_stale_cells(current_frame, max_age)
    
    def get_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current point cloud for this object."""
        return self.spatial_grid.to_point_cloud()
    
    def get_point_count(self) -> int:
        """Get the number of voxels (points) in this object."""
        return len(self.spatial_grid)
    
    def get_centroid(self) -> Optional[np.ndarray]:
        """Get the 3D centroid of this object."""
        points, _ = self.get_points()
        if len(points) == 0:
            return None
        return np.mean(points, axis=0)


class DeltaFusionSystem:
    """
    Video-compression-style 3D reconstruction system.
    
    Key differences from ObjectCentricFusion:
    - Spatial hash grid for O(1) lookups instead of linear arrays
    - Delta updates only modify changed voxels
    - Periodic keyframes ensure consistency
    - Stale voxels are cleaned up automatically
    """
    
    def __init__(self, voxel_size: float = 1.0, max_objects: int = 50,
                 expire_after_frames: int = 60, keyframe_interval: int = 30,
                 stale_voxel_age: int = 90):
        """
        Args:
            voxel_size: Size of each voxel cell in world units
            max_objects: Maximum number of tracked objects
            expire_after_frames: Remove objects not seen for this many frames
            keyframe_interval: Force keyframe every N frames per object
            stale_voxel_age: Remove voxels older than this many frames
        """
        self.voxel_size = voxel_size
        self.max_objects = max_objects
        self.expire_after_frames = expire_after_frames
        self.keyframe_interval = keyframe_interval
        self.stale_voxel_age = stale_voxel_age
        
        self.objects: Dict[int, DeltaFusionObject] = {}
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
        """Compute IoU between two 2D bounding boxes."""
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
                               new_points: Optional[np.ndarray] = None,
                               iou_threshold: float = 0.2) -> Optional[int]:
        """
        Find existing object that matches this detection using multi-factor scoring.
        
        Args:
            label: Object class name
            bbox: 2D bounding box [x1, y1, x2, y2]
            new_points: Optional 3D points for centroid matching
            iou_threshold: Minimum combined score threshold
            
        Returns:
            Object ID if match found, None otherwise
        """
        best_match_id = None
        best_score = iou_threshold
        
        # Compute new detection centroid if points provided
        new_centroid = None
        if new_points is not None and len(new_points) > 0:
            new_centroid = np.mean(new_points, axis=0)
        
        for obj_id, obj in self.objects.items():
            # Must be same class
            if obj.label != label:
                continue
            
            # Factor 1: Bbox IoU (2D spatial overlap)
            iou_score = 0.0
            if obj.bbox_history:
                iou_score = self._compute_bbox_iou(bbox, obj.bbox_history[-1])
            
            # Factor 2: Centroid distance (3D proximity)
            centroid_score = 0.0
            if new_centroid is not None:
                obj_centroid = obj.get_centroid()
                if obj_centroid is not None:
                    # Distance in world coordinates
                    dist = np.linalg.norm(new_centroid - obj_centroid)
                    # Normalize: closer = higher score (exponential decay)
                    centroid_score = np.exp(-dist / 50.0)  # 50 units = characteristic distance
            
            # Combined score: weighted average
            if new_centroid is not None:
                # Use both factors
                combined_score = 0.6 * iou_score + 0.4 * centroid_score
            else:
                # Fall back to IoU only
                combined_score = iou_score
            
            if combined_score > best_score:
                best_score = combined_score
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
        Process a frame with detected objects using delta-compression updates.
        
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
            
            pts, cols = points_per_detection[i] if i < len(points_per_detection) else (np.empty((0, 3)), np.empty((0, 3)))
            
            if len(pts) == 0:
                continue
            
            # Transform to world coordinates
            world_pts = self._transform_points_to_world(pts)
            
            # Find or create object (now with centroid distance matching)
            match_id = self._find_matching_object(label, bbox, world_pts)
            
            if match_id is not None:
                # Update existing object with delta update
                obj = self.objects[match_id]
                obj.update_points(world_pts, cols, self.frame_count)
                obj.bbox_history.append(bbox.copy())
                obj.confidence = max(obj.confidence, score)
            else:
                # Create new object
                if len(self.objects) < self.max_objects:
                    new_obj = DeltaFusionObject(
                        object_id=self.next_object_id,
                        label=label,
                        confidence=score,
                    )
                    new_obj.spatial_grid = SpatialHashGrid(voxel_size=self.voxel_size)
                    new_obj.keyframe_interval = self.keyframe_interval
                    new_obj.update_points(world_pts, cols, self.frame_count, force_keyframe=True)
                    new_obj.bbox_history.append(bbox.copy())
                    self.objects[self.next_object_id] = new_obj
                    self.next_object_id += 1
                    logging.debug(f"Created new object {new_obj.object_id}: {label}")
        
        # Cleanup phase: remove stale voxels and expire old objects
        self._cleanup(self.frame_count)
        
        # Combine all object points for rendering
        return self.get_all_points()
    
    def _cleanup(self, current_frame: int) -> None:
        """Remove stale voxels and expired objects."""
        # Clean stale voxels from each object
        for obj in self.objects.values():
            removed = obj.cleanup_stale_voxels(current_frame, self.stale_voxel_age)
            if removed > 0:
                logging.debug(f"Object {obj.object_id}: Removed {removed} stale voxels")
        
        # Expire objects not seen recently
        expired_ids = [
            obj_id for obj_id, obj in self.objects.items()
            if current_frame - obj.last_seen_frame > self.expire_after_frames
        ]
        for obj_id in expired_ids:
            logging.debug(f"Expiring object {obj_id}: {self.objects[obj_id].label}")
            del self.objects[obj_id]
    
    def get_all_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get combined point cloud from all tracked objects."""
        if not self.objects:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        
        all_pts = []
        all_cols = []
        
        for obj in self.objects.values():
            pts, cols = obj.get_points()
            if len(pts) > 0:
                all_pts.append(pts)
                all_cols.append(cols)
        
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
                "voxel_count": obj.get_point_count(),
                "frames_seen": obj.total_frames_seen,
                "last_seen": obj.last_seen_frame,
                "last_keyframe": obj.last_keyframe_frame
            }
            for obj in self.objects.values()
        ]
    
    def get_stats(self) -> Dict:
        """Get system statistics for debugging/monitoring."""
        total_voxels = sum(obj.get_point_count() for obj in self.objects.values())
        return {
            "frame_count": self.frame_count,
            "object_count": len(self.objects),
            "total_voxels": total_voxels,
            "voxel_size": self.voxel_size
        }
    
    def clear(self) -> None:
        """Reset all tracked objects."""
        self.objects.clear()
        self.next_object_id = 0
        self.frame_count = 0
        self.current_pose = np.eye(4, dtype=np.float32)
        self.last_keypoints = None
        self.last_descriptors = None
