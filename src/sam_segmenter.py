"""
SAM Segmenter Module
Handles Segment Anything Model loading and per-object mask generation.
"""
import numpy as np
import torch
import logging
from segment_anything import sam_model_registry, SamPredictor

class SAMSegmenter:
    """
    Wrapper for SAM (Segment Anything Model) that provides per-object masks
    using YOLO detection boxes as prompts.
    """
    
    def __init__(self, model_type: str = "vit_b", device: str = None):
        """
        Initialize SAM segmenter.
        
        Args:
            model_type: SAM model variant ("vit_b", "vit_l", "vit_h")
            device: Device to run on ("cuda" or "cpu"). Auto-detected if None.
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = None
        self.predictor = None
        self._current_image_embedding = None
        logging.info(f"SAM Segmenter initialized for device: {self.device}")
    
    def load_model(self, checkpoint_path: str):
        """
        Load SAM model from checkpoint.
        
        Args:
            checkpoint_path: Path to SAM checkpoint file
        """
        logging.info(f"Loading SAM model from {checkpoint_path}")
        self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        logging.info("SAM model loaded successfully")
    
    def set_image(self, image: np.ndarray):
        """
        Compute and cache the image embedding.
        Call this once per frame before querying multiple bboxes.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self.predictor.set_image(image)
        self._current_image_embedding = True
    
    def get_mask_from_bbox(self, bbox: tuple) -> np.ndarray:
        """
        Get segmentation mask for a single bounding box.
        Must call set_image() first.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Binary mask as numpy array (H, W) where True = object
        """
        if not self._current_image_embedding:
            raise RuntimeError("Must call set_image() before get_mask_from_bbox()")
        
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,  # Single best mask
        )
        
        # masks shape: (1, H, W), return as (H, W)
        return masks[0]
    
    def get_masks_for_detections(self, image: np.ndarray, boxes: np.ndarray) -> list:
        """
        Get segmentation masks for all detected objects.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            boxes: Array of bounding boxes (N, 4) as [x1, y1, x2, y2]
            
        Returns:
            List of binary masks, one per detection
        """
        if len(boxes) == 0:
            return []
        
        # Set image once, then query multiple boxes
        self.set_image(image)
        
        masks = []
        for box in boxes:
            mask = self.get_mask_from_bbox(tuple(box))
            masks.append(mask)
        
        self._current_image_embedding = None  # Clear cache
        return masks
    
    def get_combined_mask(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Get a single combined mask for all detections.
        Useful for excluding background from 3D reconstruction.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            boxes: Array of bounding boxes (N, 4)
            
        Returns:
            Combined binary mask (H, W)
        """
        masks = self.get_masks_for_detections(image, boxes)
        if not masks:
            return np.zeros(image.shape[:2], dtype=bool)
        
        combined = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined = combined | mask
        return combined
