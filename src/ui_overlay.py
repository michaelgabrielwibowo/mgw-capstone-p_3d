"""
UI Overlay Module
Provides color-coded mask overlays and user validation controls for detected objects.
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import colorsys

# Color palette for object visualization (distinct, high-contrast colors)
# Using HSV color wheel for maximum distinction
def generate_color_palette(n_colors: int = 20) -> List[Tuple[int, int, int]]:
    """
    Generate a palette of visually distinct colors.
    
    Args:
        n_colors: Number of colors to generate
        
    Returns:
        List of BGR color tuples
    """
    colors = []
    for i in range(n_colors):
        hue = (i * 137.5) % 360  # Golden angle approximation for good distribution
        sat = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        val = 0.9 - (i % 2) * 0.1  # Vary value slightly
        
        # HSV to RGB (0-1 range)
        r, g, b = colorsys.hsv_to_rgb(hue / 360, sat, val)
        
        # Convert to BGR (0-255 range) for OpenCV
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    
    return colors

# Pre-generated color palette
COLOR_PALETTE = generate_color_palette(20)


class UIOverlay:
    """
    Manages UI overlays for object visualization and user interaction.
    """
    
    def __init__(self):
        self.colors = COLOR_PALETTE
        self.selected_object_idx = -1  # -1 = no selection
        self.mode = "view"  # "view", "add_object", "edit"
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        self.object_states: Dict[int, str] = {}  # idx -> "accepted" / "rejected" / "pending"
        
    def get_color(self, idx: int) -> Tuple[int, int, int]:
        """Get color for object at index."""
        return self.colors[idx % len(self.colors)]
    
    def draw_mask_overlay(self, frame: np.ndarray, mask: np.ndarray, 
                          color: Tuple[int, int, int], alpha: float = 0.4) -> np.ndarray:
        """
        Draw semi-transparent colored mask overlay on frame.
        
        Args:
            frame: BGR image
            mask: Boolean mask (H, W)
            color: BGR color tuple
            alpha: Transparency (0-1)
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        overlay[mask] = color
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def draw_label(self, frame: np.ndarray, text: str, position: Tuple[int, int],
                   bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Draw label with background at position.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = position
        padding = 4
        cv2.rectangle(frame, 
                      (x - padding, y - text_height - padding),
                      (x + text_width + padding, y + padding),
                      bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return frame
    
    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """Draw bounding box on frame."""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
    
    def render_detections(self, frame: np.ndarray, 
                          detections: List[Tuple[np.ndarray, str, float, np.ndarray]],
                          show_masks: bool = True,
                          show_labels: bool = True,
                          show_bboxes: bool = True,
                          confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Render all detections with color-coded overlays.
        
        Args:
            frame: BGR image
            detections: List of (mask, label, confidence, bbox)
            show_masks: Whether to show mask overlays
            show_labels: Whether to show labels
            show_bboxes: Whether to show bounding boxes
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Frame with overlays
        """
        overlay_frame = frame.copy()
        
        for idx, (mask, label, score, bbox) in enumerate(detections):
            if score < confidence_threshold:
                continue
                
            color = self.get_color(idx)
            state = self.object_states.get(idx, "pending")
            
            # Modify color based on state
            if state == "accepted":
                # Add green tint
                color = (color[0], min(255, color[1] + 50), color[2])
            elif state == "rejected":
                # Dim the color
                color = tuple(c // 2 for c in color)
            
            # Draw mask overlay
            if show_masks and mask is not None:
                overlay_frame = self.draw_mask_overlay(overlay_frame, mask, color, alpha=0.3)
            
            # Draw bounding box
            if show_bboxes:
                thickness = 3 if idx == self.selected_object_idx else 2
                overlay_frame = self.draw_bbox(overlay_frame, bbox, color, thickness)
            
            # Draw label
            if show_labels:
                x1, y1, x2, y2 = map(int, bbox)
                label_text = f"{label} {score:.0%}"
                if state == "accepted":
                    label_text += " ✓"
                elif state == "rejected":
                    label_text += " ✗"
                overlay_frame = self.draw_label(overlay_frame, label_text, (x1, y1 - 5), color)
        
        return overlay_frame
    
    def draw_controls_help(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard controls help on frame."""
        help_lines = [
            "Controls:",
            "[Q] Quit",
            "[M] Toggle masks",
            "[L] Toggle labels",
            "[B] Toggle boxes",
            "[A] Accept selected",
            "[R] Reject selected",
            "[Tab] Next object",
            "[Space] Add object mode"
        ]
        
        y_offset = 20
        for line in help_lines:
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        # Draw mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(frame, mode_text, (frame.shape[1] - 150, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def draw_add_object_box(self, frame: np.ndarray) -> np.ndarray:
        """Draw the box being drawn in add_object mode."""
        if self.drawing_box and self.box_start and self.box_end:
            cv2.rectangle(frame, self.box_start, self.box_end, (0, 255, 0), 2)
            cv2.putText(frame, "Drawing new object...", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
    
    def handle_mouse_event(self, event: int, x: int, y: int, flags: int):
        """Handle mouse events for object selection and drawing."""
        if self.mode == "add_object":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing_box = True
                self.box_start = (x, y)
                self.box_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_box:
                self.box_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing_box = False
                self.box_end = (x, y)
    
    def get_drawn_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the user-drawn box as (x1, y1, x2, y2) or None if not complete."""
        if self.box_start and self.box_end and not self.drawing_box:
            x1 = min(self.box_start[0], self.box_end[0])
            y1 = min(self.box_start[1], self.box_end[1])
            x2 = max(self.box_start[0], self.box_end[0])
            y2 = max(self.box_start[1], self.box_end[1])
            
            # Reset box
            self.box_start = None
            self.box_end = None
            
            # Only return if box is big enough
            if x2 - x1 > 10 and y2 - y1 > 10:
                return (x1, y1, x2, y2)
        return None
    
    def accept_object(self, idx: int):
        """Mark object as accepted."""
        self.object_states[idx] = "accepted"
    
    def reject_object(self, idx: int):
        """Mark object as rejected."""
        self.object_states[idx] = "rejected"
    
    def toggle_mode(self, new_mode: str):
        """Switch UI mode."""
        self.mode = new_mode
        self.box_start = None
        self.box_end = None
        self.drawing_box = False
    
    def select_next_object(self, total_objects: int):
        """Select the next object in the list."""
        if total_objects > 0:
            self.selected_object_idx = (self.selected_object_idx + 1) % total_objects
