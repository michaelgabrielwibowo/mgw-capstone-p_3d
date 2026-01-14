import unittest
import os
import sys

from src.main import VisionSystem
from src import config

class TestVisionSystem(unittest.TestCase):
    def test_load_models(self):
        """
        Tests if the models are loaded correctly.
        """
        # Create a VisionSystem instance
        vision_system = VisionSystem()

        # Check if the models are not loaded initially
        self.assertIsNone(vision_system.yolo)
        self.assertIsNone(vision_system.depth_model)

        # Load the models
        vision_system.load_models()

        # Check if the models are loaded
        self.assertIsNotNone(vision_system.yolo)
        self.assertIsNotNone(vision_system.depth_model)

if __name__ == '__main__':
    unittest.main()
