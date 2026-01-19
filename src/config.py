import os
import yaml

# -- Project Root --
# Dynamically determine the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load config from YAML file
with open(os.path.join(PROJECT_ROOT, 'config.yml'), 'r') as f:
    config = yaml.safe_load(f)

# -- Model Paths --
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, config['YOLO_MODEL_PATH'])
DEPTH_MODEL_PATH = os.path.join(PROJECT_ROOT, config['DEPTH_MODEL_PATH'])
SAM_MODEL_PATH = os.path.join(PROJECT_ROOT, config['SAM_MODEL_PATH'])
SAM_MODEL_TYPE = config['SAM_MODEL_TYPE']

# -- 3D Reconstruction --
FOCAL_LENGTH = config['FOCAL_LENGTH']
DEPTH_SCALE = config['DEPTH_SCALE']

# -- n8n Integration --
N8N_WEBHOOK_URL = config['N8N_WEBHOOK_URL']

# -- Camera --
CAMERA_INDEX = config['CAMERA_INDEX']
