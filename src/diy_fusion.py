import numpy as np
import cv2

class DIYFusionSystem:
    def __init__(self, voxel_size=0.05): # voxel_size=0.05 means 5cm resolution
        # The global "World" map
        self.global_points = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.uint8)
        self.voxel_size = voxel_size
        
        # Tracking (Odometry) variables
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.last_keypoints = None
        self.last_descriptors = None
        # Camera Pose: Starts at (0,0,0) facing forward
        self.current_pose = np.eye(4) 
        
        # Camera intrinsics (Set on first frame)
        self.K = None

    def estimate_motion(self, frame):
        """
        Calculates how much the camera moved between frames using ORB features.
        Returns: True if motion was calculated, False if tracking was lost.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if self.last_keypoints is None:
            self.last_keypoints, self.last_descriptors = keypoints, descriptors
            return True # First frame, valid but no motion yet

        # Match features between frames
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.last_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # We need at least 8 good matches to calculate generic motion
        if len(matches) < 8:
            return False

        # Extract coordinates of matching points
        pts1 = np.float32([self.last_keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

        # Calculate Essential Matrix and Recover Pose
        # Note: In monocular vision, we don't know the absolute scale, 
        # so we assume unit scale (1.0).
        E, mask = cv2.findEssentialMat(pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return False
            
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K)

        # Update global pose: T_new = T_old * T_relative
        T_relative = np.eye(4)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.reshape(3)
        
        self.current_pose = self.current_pose @ T_relative

        # Update history for next frame
        self.last_keypoints, self.last_descriptors = keypoints, descriptors
        return True

    def process_frame(self, points_3d, colors, frame, camera_matrix):
        """
        Takes the NEW points from the current frame, rotates them to match the
        camera's current position, and adds them to the global world.
        """
        if self.K is None:
            self.K = camera_matrix

        # 1. Update Camera Position (Tracking)
        if not self.estimate_motion(frame):
            # If tracking fails, we skip adding points this frame to avoid "garbage" data
            return self.global_points, self.global_colors

        # 2. Transform new points to Global Space
        # Formula: P_global = R * P_local + t
        R = self.current_pose[:3, :3]
        t = self.current_pose[:3, 3]
        
        # Reshape for matrix multiplication
        # points_3d is (N, 3), we need (3, N) for rotation, then transpose back
        points_rotated = (R @ points_3d.T).T + t

        # 3. Add to Global Map
        self.global_points = np.vstack((self.global_points, points_rotated))
        self.global_colors = np.vstack((self.global_colors, colors))

        # 4. Voxel Downsampling (The "Memory Saver")
        # Prevents memory from exploding by keeping only 1 point per voxel cube.
        # We only filter when the cloud gets too large (e.g., > 10,000 points)
        if len(self.global_points) > 10000: 
            self.global_points, self.global_colors = self._voxel_filter(
                self.global_points, self.global_colors, self.voxel_size
            )

        return self.global_points, self.global_colors

    def _voxel_filter(self, points, colors, voxel_size):
        """
        A pure NumPy implementation of voxel grid downsampling.
        """
        # Snap points to grid
        quantized = (points / voxel_size).astype(np.int32)
        
        # Find unique voxels
        # unique_indices keeps the index of the *first* point found in each voxel
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        
        return points[unique_indices], colors[unique_indices]