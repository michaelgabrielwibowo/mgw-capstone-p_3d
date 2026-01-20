"""
Asynchronous SAM Processor

Runs SAM (Segment Anything Model) inference in a background thread
to prevent blocking the main processing loop.
"""

import threading
import queue
from typing import Optional, Tuple, List
import numpy as np
import logging


class AsyncSAMProcessor:
    """
    Background thread processor for SAM inference.
    
    Uses double-buffering to prevent blocking:
    - Main thread submits work to input queue (non-blocking)
    - Worker thread processes SAM inference
    - Main thread retrieves results from output queue (non-blocking)
    """
    
    def __init__(self, sam_segmenter):
        """
        Args:
            sam_segmenter: SAMSegmenter instance
        """
        self.sam = sam_segmenter
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logging.info("AsyncSAMProcessor started")
    
    def _worker(self):
        """Background worker thread that processes SAM inference."""
        while self.running:
            try:
                # Wait for work (blocking with timeout)
                work_item = self.input_queue.get(timeout=0.1)
                if work_item is None:
                    break  # Shutdown signal
                
                frame, boxes = work_item
                
                # Run SAM inference (heavy operation)
                try:
                    masks = self.sam.get_masks_for_detections(frame, boxes)
                    
                    # Put result in output queue (discard old results)
                    try:
                        self.output_queue.put_nowait(masks)
                    except queue.Full:
                        # Discard old result, put new one
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.output_queue.put_nowait(masks)
                
                except Exception as e:
                    logging.error(f"SAM inference error: {e}")
            
            except queue.Empty:
                # No work available, continue
                continue
            except Exception as e:
                logging.error(f"AsyncSAM worker error: {e}")
        
        logging.info("AsyncSAMProcessor stopped")
    
    def submit(self, frame: np.ndarray, boxes: np.ndarray) -> bool:
        """
        Submit a frame for SAM processing (non-blocking).
        
        Args:
            frame: RGB image
            boxes: Bounding boxes from YOLO
            
        Returns:
            True if work was submitted, False if queue was full
        """
        try:
            self.input_queue.put_nowait((frame, boxes))
            return True
        except queue.Full:
            # Already processing, skip this frame
            return False
    
    def get_result(self) -> Optional[List]:
        """
        Try to get SAM results (non-blocking).
        
        Returns:
            List of masks if available, None otherwise
        """
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def shutdown(self):
        """Stop the background thread."""
        self.running = False
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)
