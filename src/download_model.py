import os
from ultralytics import YOLO
import config

def download_and_optimize_model():
    """
    Downloads the YOLOv8n model and exports it to ONNX format if the files don't already exist.
    """
    yolo_onnx_path = config.YOLO_MODEL_PATH
    yolo_pt_path = yolo_onnx_path.replace('.onnx', '.pt')

    if os.path.exists(yolo_pt_path) and os.path.exists(yolo_onnx_path):
        print("YOLOv8n models already exist. Skipping download and export.")
        return

    if not os.path.exists(yolo_pt_path):
        print("Downloading YOLOv8n model...")
        model = YOLO(yolo_pt_path)
        print("Model downloaded successfully!")
    else:
        model = YOLO(yolo_pt_path)

    if not os.path.exists(yolo_onnx_path):
        print("Exporting to ONNX format for optimization...")
        model.export(format='onnx')
        print(f"Model exported to ONNX format as '{yolo_onnx_path}'")
    
    # Optionally export to OpenVINO format for Intel hardware optimization
    try:
        openvino_path = yolo_onnx_path.replace('.onnx', '_openvino_model')
        if not os.path.exists(openvino_path):
            print("Exporting to OpenVINO format for Intel hardware optimization...")
            model.export(format='openvino')
            print("Model exported to OpenVINO format")
        else:
            print("OpenVINO model already exists. Skipping export.")
    except Exception as e:
        print(f"Could not export to OpenVINO: {e}")
        print("This is OK - continuing with ONNX model")



if __name__ == "__main__":
    download_and_optimize_model()