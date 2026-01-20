"""
Point Cloud Export Utilities

Provides functions to export 3D point clouds to standard formats.
"""

import numpy as np
from typing import Tuple
import logging


def export_ply(filename: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Export point cloud to PLY format.
    
    Args:
        filename: Output filename (e.g. "scene.ply")
        points: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors (0-255)
    """
    if len(points) == 0:
        logging.warning("Cannot export empty point cloud")
        return
    
    if len(points) != len(colors):
        raise ValueError(f"Points ({len(points)}) and colors ({len(colors)}) must have same length")
    
    try:
        with open(filename, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write point data
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        
        logging.info(f"Exported {len(points)} points to {filename}")
    
    except Exception as e:
        logging.error(f"Failed to export PLY: {e}")
        raise


def export_ply_binary(filename: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Export point cloud to binary PLY format (faster, smaller files).
    
    Args:
        filename: Output filename (e.g. "scene.ply")
        points: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors (0-255)
    """
    if len(points) == 0:
        logging.warning("Cannot export empty point cloud")
        return
    
    if len(points) != len(colors):
        raise ValueError(f"Points ({len(points)}) and colors ({len(colors)}) must have same length")
    
    try:
        import struct
        
        with open(filename, 'wb') as f:
            # PLY header (still ASCII)
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            f.write(header.encode('ascii'))
            
            # Write binary data
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                # Pack as: 3 floats + 3 unsigned chars
                data = struct.pack('<fffBBB', x, y, z, int(r), int(g), int(b))
                f.write(data)
        
        logging.info(f"Exported {len(points)} points to {filename} (binary)")
    
    except Exception as e:
        logging.error(f"Failed to export binary PLY: {e}")
        raise
