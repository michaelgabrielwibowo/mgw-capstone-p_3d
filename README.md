# Real-Time 3D Vision System

This project implements a real-time 3D vision system using YOLO for object detection and Depth Anything V2 for depth estimation.

## Architecture

```
Laptop Camera / Phone Stream
    ↓
Vision Pipeline (Python: YOLO + Depth Anything V2)
    ↓
3D Point Cloud Visualization
```

## Setup Instructions

### 1. Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Download Models

1. Download the YOLO model:
   ```bash
   python download_model.py
   ```
2. Download the Depth Anything V2 model:
    ```bash
    python download_depth_anything_v2.py
    ```

### 3. Run the Vision System

1. Run the main vision system with 3D visualization:
   ```bash
   python main.py
   ```

## Configuration

- The `config.py` file contains the configuration for the application, including model paths, camera index, and 3D reconstruction parameters.

## Components

- **Vision Core**: YOLOv8n for object detection and Depth Anything V2 for depth estimation.
- **3D Visualization**: Real-time 3D point cloud visualization using PyVista.