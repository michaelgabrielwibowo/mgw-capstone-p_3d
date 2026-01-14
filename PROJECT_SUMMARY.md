# Real-Time 3D Vision System

## Project Overview

This project implements a real-time 3D vision system that converts 2D camera feeds into 3D representations using depth estimation and computer vision techniques. The system combines:

- YOLO for object detection
- Depth Anything V2 for depth estimation
- PyVista for 3D visualization

## Hardware Requirements

- Intel i5-1355U (or similar multi-core processor)
- 12 GB RAM minimum
- Dedicated or integrated GPU (Iris Xe or better)
- USB or built-in camera
- Windows/Linux/macOS operating system

## Setup Instructions

### 1. Prerequisites

1. Install Python 3.8 or higher
2. Install Git (for downloading models)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models
```bash
python download_model.py
python download_depth_anything_v2.py
```

## How to Run the System

1. Open a terminal/command prompt
2. Navigate to the capstone_project directory
3. Run: `python main.py`
4. Point your camera at objects to see 3D reconstruction with object detection overlays

## How It Works

### 1. Vision Pipeline
- Captures real-time video from camera
- Uses YOLOv8n to detect objects in each frame
- Applies Depth Anything V2 depth estimation to create depth maps
- Combines 2D image data with depth information

### 2. 3D Reconstruction
- Converts 2D pixels + depth to 3D coordinates
- Creates point cloud representation of scene
- Maps colors from original image to 3D points
- Visualizes in PyVista 3D viewer

## Performance Expectations

- Vision processing: 3.5–6 FPS (detections)
- 3D Projection: <300ms latency

## Troubleshooting

### Common Issues:

1. **Camera not opening**: Check if another application is using the camera
2. **Slow performance**: Reduce input resolution or use smaller models
3. **Memory issues**: Close other applications to free up RAM
4. **Model download errors**: Check internet connection and retry

### Performance Optimization:

1. Use ONNX-optimized models for faster inference
2. Reduce input resolution for faster processing
3. Limit point cloud density for better visualization performance

## Files Included

- `main.py`: Main script for the 3D vision system
- `config.py`: Configuration file for the system
- `download_model.py`: Model download and optimization for YOLO
- `download_depth_anything_v2.py`: Model download for Depth Anything V2

## Future Enhancements

1. Implement NeRF for higher-quality 3D reconstruction
2. Add multi-view geometry for better depth estimation
3. Integrate more advanced scene understanding models
4. Add AR overlay capabilities
5. Implement real-time mesh generation

## License

This project is created for educational purposes as part of a capstone project.