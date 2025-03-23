#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic tests for Coin Meter
"""

import os
import sys
import unittest

# Add parent directory to path to import coin_meter modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.camera import Camera, CameraCalibration
from src.detector import CoinDetector
from src.models import CoinModel
from src.utils import setup_logging, generate_summary, calculate_total_value, format_coin_value, list_available_cameras

class TestBasic(unittest.TestCase):
    """Basic tests for Coin Meter"""
    
    def setUp(self):
        """Set up test environment"""
        # Setup logging
        setup_logging(verbose=False)
        
        # Define directories
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.models_dir = os.path.join(self.base_dir, 'models')
        
    def test_coin_model_loading(self):
        """Test loading coin models"""
        # Initialize model
        coin_model = CoinModel(self.models_dir)
        
        # Get coin types
        coin_types = coin_model.get_coin_types()
        
        # Check if we have any models
        self.assertTrue(len(coin_types) > 0, "No coin models found")
        
        # Check if we can get data for each coin type
        for coin_type in coin_types:
            data = coin_model.get_coin_data(coin_type)
            self.assertIsNotNone(data, f"Failed to get data for {coin_type}")
            self.assertEqual(data['coin_type'], coin_type, f"Coin type mismatch for {coin_type}")
            
    def test_list_cameras(self):
        """Test listing available cameras"""
        # This test just checks if the function runs without errors
        # It doesn't verify the actual cameras since that depends on the system
        cameras = list_available_cameras()
        self.assertIsInstance(cameras, list, "list_available_cameras should return a list")
        # Each camera should be a tuple of (id, name)
        for camera in cameras:
            self.assertIsInstance(camera, tuple, "Each camera should be a tuple")
            self.assertEqual(len(camera), 2, "Each camera tuple should have 2 elements")
            self.assertIsInstance(camera[0], int, "Camera ID should be an integer")
            self.assertIsInstance(camera[1], str, "Camera name should be a string")
    
    def test_utils_functions(self):
        """Test utility functions"""
        # Test format_coin_value
        self.assertEqual(format_coin_value('1_cent'), '1 Cent')
        self.assertEqual(format_coin_value('2_euro'), '2 Euro')
        self.assertEqual(format_coin_value('unknown'), 'unknown')
        
        # Test calculate_total_value
        coins = [
            {'type': '1_cent'},
            {'type': '2_cent'},
            {'type': '1_euro'},
            {'type': '2_euro'}
        ]
        total = calculate_total_value(coins)
        self.assertEqual(total, 3.03, "Total value calculation incorrect")
        
        # Test generate_summary
        summary = generate_summary(coins)
        self.assertEqual(summary['total_coins'], 4, "Total coins count incorrect")
        self.assertEqual(summary['coin_counts']['1_cent'], 1, "1 cent count incorrect")
        self.assertEqual(summary['coin_counts']['2_euro'], 1, "2 euro count incorrect")
        self.assertEqual(summary['total_value_eur'], 3.03, "Total value incorrect")
        self.assertEqual(summary['formatted_value'], "3.03 €", "Formatted value incorrect")
        
    def test_detector_initialization(self):
        """Test detector initialization"""
        # Initialize model
        coin_model = CoinModel(self.models_dir)
        
        # Initialize detector
        detector = CoinDetector(coin_model)
        
        # Check detector attributes
        self.assertIsNotNone(detector.coin_model, "Detector coin model is None")
        self.assertEqual(detector.coin_model, coin_model, "Detector coin model mismatch")

if __name__ == '__main__':
    unittest.main()
