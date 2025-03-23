#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic usage example for Coin Meter
"""

import os
import sys
import json

# Add parent directory to path to import coin_meter modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.camera import Camera, CameraCalibration
from src.detector import CoinDetector
from src.models import CoinModel
from src.utils import setup_logging, generate_summary

def main():
    """Main function demonstrating basic usage"""
    # Setup logging
    setup_logging(verbose=True)
    
    # Define directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    calibration_dir = os.path.join(base_dir, 'calibration')
    pictures_dir = os.path.join(base_dir, 'pictures')
    models_dir = os.path.join(base_dir, 'models')
    
    # Ensure directories exist
    for directory in [calibration_dir, pictures_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Example 0: List available cameras
    print("\n=== Example 0: List Available Cameras ===")
    print("To list all available cameras:")
    print("1. Run the list_cameras command:")
    print(f"   python -m src.main list_cameras")
    
    # Try to list cameras directly
    from src.utils import list_available_cameras
    cameras = list_available_cameras()
    if cameras:
        print(f"\nFound {len(cameras)} camera(s):")
        for camera_id, camera_name in cameras:
            print(f"  Camera ID: {camera_id} - {camera_name}")
    else:
        print("\nNo cameras found or OpenCV not installed")
    
    # Example 1: Camera calibration
    print("\n=== Example 1: Camera Calibration ===")
    print("Initializing camera...")
    camera = Camera(camera_id=0, width=1280, height=720)
    
    calibration = CameraCalibration()
    calibration_file = os.path.join(calibration_dir, 'camera_calibration.json')
    coin_calibration_file = os.path.join(calibration_dir, 'coin_calibration.json')
    
    if os.path.exists(calibration_file):
        print(f"Loading existing calibration from {calibration_file}")
        calibration.load(calibration_file)
    else:
        print("No calibration file found. To calibrate:")
        print("Option 1: Using a chessboard pattern")
        print(f"   python -m src.main calibrate --camera-id 0 --resolution 1280x720 --output {calibration_file}")
        print("   Show a 9x6 chessboard pattern to the camera")
        print("\nOption 2: Using a 2 Euro coin as reference")
        print(f"   python -m src.main calibrate-with-coin --camera-id 0 --resolution 1280x720 --output {coin_calibration_file}")
        print("   Place a 2 Euro coin in the camera view")
    
    # Example 2: Capture an image
    print("\n=== Example 2: Image Capture ===")
    print("To capture images:")
    print("1. Run the capture command:")
    print(f"   python -m src.main capture --camera-id 0 --resolution 1280x720 --output-dir {pictures_dir}")
    print("2. Press 'c' to capture, 'q' to quit")
    
    # Example 3: Train a model
    print("\n=== Example 3: Model Training ===")
    print("To train a model for a specific coin:")
    print("1. Run the training command:")
    print(f"   python -m src.main train-from-camera --camera-id 0 --resolution 1280x720 --coin-type 1_euro --output-dir {models_dir}")
    print("2. Place the coin in the camera view")
    print("3. Press 'c' to capture training data, 'q' to finish")
    
    # Example 4: Detect coins in an image
    print("\n=== Example 4: Coin Detection ===")
    
    # Check if we have any models
    if os.path.exists(models_dir) and any(f.endswith('.json') for f in os.listdir(models_dir)):
        print("Loading coin models...")
        coin_model = CoinModel(models_dir)
        
        # Initialize detector with calibration if available
        if os.path.exists(coin_calibration_file):
            print(f"Using coin calibration from {coin_calibration_file}")
            calibration = CameraCalibration()
            calibration.load(coin_calibration_file)
            detector = CoinDetector(coin_model, calibration)
            
            if calibration.get_pixels_per_mm():
                print(f"Calibration loaded: {calibration.get_pixels_per_mm():.2f} pixels per mm")
        else:
            detector = CoinDetector(coin_model)
        
        # Check if we have any images
        if os.path.exists(pictures_dir) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(pictures_dir)):
            image_files = [f for f in os.listdir(pictures_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} images in {pictures_dir}")
            
            if image_files:
                example_image = os.path.join(pictures_dir, image_files[0])
                print(f"Detecting coins in {example_image}...")
                
                coins = detector.detect(example_image, save_debug=True)
                
                if coins:
                    print(f"Found {len(coins)} coins:")
                    for coin in coins:
                        if 'diameter_mm' in coin:
                            print(f"  {coin['type']}: {coin['confidence']:.2f} confidence, {coin['diameter_mm']:.1f}mm diameter")
                        else:
                            print(f"  {coin['type']}: {coin['confidence']:.2f} confidence")
                    
                    # Generate summary
                    summary = generate_summary(coins)
                    print(f"\nTotal value: {summary['formatted_value']}")
                else:
                    print("No coins detected")
        else:
            print("No images found. Capture some images first.")
    else:
        print("No coin models found. Train some models first.")
    
    print("\n=== Complete Example ===")
    print("For a complete workflow:")
    print("1. Calibrate the camera")
    print("2. Train models for each coin type")
    print("3. Capture images of coins")
    print("4. Detect coins in the images")
    print("\nSee the README.md file for more details")

if __name__ == "__main__":
    main()
