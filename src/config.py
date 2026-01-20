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

# -- Delta Fusion Settings --
DELTA_FUSION_CONFIG = config.get('DELTA_FUSION', {})
VOXEL_SIZE = DELTA_FUSION_CONFIG.get('voxel_size', 2.5)
MAX_OBJECTS = DELTA_FUSION_CONFIG.get('max_objects', 20)
EXPIRE_AFTER_FRAMES = DELTA_FUSION_CONFIG.get('expire_after_frames', 45)
KEYFRAME_INTERVAL = DELTA_FUSION_CONFIG.get('keyframe_interval', 20)
STALE_VOXEL_AGE = DELTA_FUSION_CONFIG.get('stale_voxel_age', 60)
SAM_SKIP_FRAMES = DELTA_FUSION_CONFIG.get('sam_skip_frames', 3)
RENDER_SKIP_FRAMES = DELTA_FUSION_CONFIG.get('render_skip_frames', 2)
BLEND_MODE = DELTA_FUSION_CONFIG.get('blend_mode', 'average')
TEMPORAL_SMOOTHING_WEIGHT = DELTA_FUSION_CONFIG.get('temporal_smoothing_weight', 0.3)
