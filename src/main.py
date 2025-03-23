#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coin Meter - CLI-based coin recognition program
"""

import os
import sys
import json
import cv2
import numpy as np
import click
from datetime import datetime

# Import local modules
try:
    # When running as a module (python -m src.main)
    from src.camera import Camera, CameraCalibration
    from src.detector import CoinDetector
    from src.models import CoinModel
    from src.utils import setup_logging, ensure_directories, list_available_cameras
except ImportError:
    # When running directly (python coin_meter.py)
    from camera import Camera, CameraCalibration
    from detector import CoinDetector
    from models import CoinModel
    from utils import setup_logging, ensure_directories, list_available_cameras

# Define the version
VERSION = "0.1.0"

# Define the default directories
DEFAULT_CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration")
DEFAULT_PICTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pictures")
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Ensure the directories exist
ensure_directories([DEFAULT_CALIBRATION_DIR, DEFAULT_PICTURES_DIR, DEFAULT_MODELS_DIR])

@click.group()
@click.version_option(version=VERSION)
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Coin Meter - A CLI tool for recognizing Euro coins using a webcam."""
    # Initialize the context object
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    setup_logging(verbose)

@cli.command()
@click.pass_context
def list_cameras(ctx):
    """List all available cameras."""
    cameras = list_available_cameras()
    if cameras:
        click.echo(f"Found {len(cameras)} camera(s):")
        for camera_id, camera_name in cameras:
            click.echo(f"  Camera ID: {camera_id} - {camera_name}")
    else:
        click.echo("No cameras found")

@cli.command()
@click.option('--camera-id', default=0, help='Camera ID to use')
@click.option('--resolution', default='1280x720', help='Camera resolution (WIDTHxHEIGHT)')
@click.option('--output', default=os.path.join(DEFAULT_CALIBRATION_DIR, 'camera_calibration.json'), 
              help='Output file for calibration data')
@click.pass_context
def calibrate(ctx, camera_id, resolution, output):
    """Calibrate the camera using a chessboard pattern."""
    width, height = map(int, resolution.split('x'))
    
    # Initialize the camera
    camera = Camera(camera_id, width, height)
    
    # Initialize the calibration
    calibration = CameraCalibration()
    
    # Run the calibration
    click.echo("Starting camera calibration...")
    calibration.calibrate(camera)
    
    # Save the calibration data
    calibration.save(output)
    click.echo(f"Calibration data saved to {output}")

@cli.command(name='calibrate-with-coin')
@click.option('--camera-id', default=0, help='Camera ID to use')
@click.option('--resolution', default='1280x720', help='Camera resolution (WIDTHxHEIGHT)')
@click.option('--coin-type', default='2_euro', help='Coin type to use as reference')
@click.option('--output', default=os.path.join(DEFAULT_CALIBRATION_DIR, 'coin_calibration.json'), 
              help='Output file for calibration data')
@click.option('--num-samples', default=5, help='Number of samples to collect')
@click.pass_context
def calibrate_with_coin(ctx, camera_id, resolution, coin_type, output, num_samples):
    """Calibrate the camera using a 2 Euro coin as reference."""
    width, height = map(int, resolution.split('x'))
    
    # Initialize the camera
    camera = Camera(camera_id, width, height)
    
    # Initialize the calibration
    calibration = CameraCalibration()
    
    # Run the calibration
    click.echo(f"Starting camera calibration using a {coin_type} coin as reference...")
    click.echo(f"Place a {coin_type} coin in the camera view.")
    click.echo(f"Need {num_samples} good samples.")
    
    calibration.calibrate_with_coin(camera, coin_type=coin_type, num_samples=num_samples)
    
    # Save the calibration data
    calibration.save(output)
    click.echo(f"Calibration data saved to {output}")

@cli.command()
@click.option('--camera-id', default=0, help='Camera ID to use')
@click.option('--resolution', default='1280x720', help='Camera resolution (WIDTHxHEIGHT)')
@click.option('--brightness', default=None, type=int, help='Camera brightness')
@click.option('--contrast', default=None, type=int, help='Camera contrast')
@click.option('--exposure', default=None, type=int, help='Camera exposure')
@click.option('--light-compensation', is_flag=True, help='Enable automatic histogram equalization')
@click.option('--output-dir', default=DEFAULT_PICTURES_DIR, help='Directory to save captured images')
@click.pass_context
def capture(ctx, camera_id, resolution, brightness, contrast, exposure, light_compensation, output_dir):
    """Capture images from the camera."""
    width, height = map(int, resolution.split('x'))
    
    # Initialize the camera
    camera = Camera(camera_id, width, height)
    
    # Set camera properties if provided
    if brightness is not None:
        camera.set_brightness(brightness)
    if contrast is not None:
        camera.set_contrast(contrast)
    if exposure is not None:
        camera.set_exposure(exposure)
    
    # Enable light compensation if requested
    if light_compensation:
        camera.enable_light_compensation()
    
    # Capture images
    click.echo("Starting image capture. Press 'c' to capture, 'q' to quit.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"image_{timestamp}.jpg")
    
    camera.capture_interactive(filename)
    click.echo(f"Image saved to {filename}")

@cli.command()
@click.option('--input', required=True, help='Input image or directory of images')
@click.option('--model-dir', default=DEFAULT_MODELS_DIR, help='Directory containing model data')
@click.option('--calibration', default=None, help='Calibration file to use for size measurements')
@click.option('--output', default=None, help='Output file for detection results')
@click.option('--save-debug', is_flag=True, help='Save debug images with contours')
@click.pass_context
def detect(ctx, input, model_dir, calibration, output, save_debug):
    """Detect coins in images."""
    # Load models
    coin_model = CoinModel(model_dir)
    
    # Initialize detector
    detector = CoinDetector(coin_model)
    
    # Load calibration if provided
    if calibration:
        calibration_path = os.path.abspath(calibration)
        if os.path.exists(calibration_path):
            click.echo(f"Using calibration file: {calibration_path}")
            calibration_obj = CameraCalibration()
            if calibration_obj.load(calibration_path):
                detector.calibration = calibration_obj
                if calibration_obj.get_pixels_per_mm():
                    click.echo(f"Calibration loaded: {calibration_obj.get_pixels_per_mm():.2f} pixels per mm")
                else:
                    click.echo("Calibration loaded but no pixels per mm data found")
            else:
                click.echo(f"Failed to load calibration file: {calibration_path}")
        else:
            click.echo(f"Calibration file not found: {calibration_path}")
    
    # Process input (file or directory)
    if os.path.isfile(input):
        files = [input]
    elif os.path.isdir(input):
        files = [os.path.join(input, f) for f in os.listdir(input) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        click.echo(f"Error: Input '{input}' is not a valid file or directory")
        return
    
    # Process each file
    results = {}
    for file in files:
        click.echo(f"Processing {file}...")
        coins = detector.detect(file, save_debug=save_debug, calibration_file=calibration)
        
        # Print results
        click.echo(f"Found {len(coins)} coins:")
        for coin in coins:
            click.echo(f"  {coin['type']}: {coin['confidence']:.2f} confidence")
        
        # Store results
        results[os.path.basename(file)] = {
            "image": os.path.basename(file),
            "coins": coins
        }
    
    # Save results if output is specified
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {output}")

@cli.command(name='train-from-camera')
@click.option('--camera-id', default=0, help='Camera ID to use')
@click.option('--resolution', default='1280x720', help='Camera resolution (WIDTHxHEIGHT)')
@click.option('--coin-type', required=True, 
              type=click.Choice(['1_cent', '2_cent', '5_cent', '10_cent', '20_cent', '50_cent', '1_euro', '2_euro']),
              help='Type of coin to train')
@click.option('--output-dir', default=DEFAULT_MODELS_DIR, help='Directory to save model data')
@click.pass_context
def train_from_camera(ctx, camera_id, resolution, coin_type, output_dir):
    """Train the model using the camera."""
    width, height = map(int, resolution.split('x'))
    
    # Initialize the camera
    camera = Camera(camera_id, width, height)
    
    # Initialize the model
    coin_model = CoinModel(output_dir)
    
    # Train the model
    click.echo(f"Training model for {coin_type}...")
    click.echo("Place the coin in the camera view and press 'c' to capture training data.")
    click.echo("Press 'q' to finish training.")
    
    coin_model.train_from_camera(camera, coin_type)
    
    # Save the model
    output_file = os.path.join(output_dir, f"{coin_type}.json")
    coin_model.save(coin_type, output_file)
    click.echo(f"Model for {coin_type} saved to {output_file}")

@cli.command(name='detect-live')
@click.option('--camera-id', default=0, help='Camera ID to use')
@click.option('--resolution', default='1280x720', help='Camera resolution (WIDTHxHEIGHT)')
@click.option('--model-dir', default=DEFAULT_MODELS_DIR, help='Directory containing model data')
@click.option('--calibration', default=None, help='Calibration file to use for size measurements')
@click.option('--brightness', default=None, type=int, help='Camera brightness')
@click.option('--contrast', default=None, type=int, help='Camera contrast')
@click.option('--exposure', default=None, type=int, help='Camera exposure')
@click.option('--light-compensation', is_flag=True, help='Enable automatic histogram equalization')
@click.pass_context
def detect_live(ctx, camera_id, resolution, model_dir, calibration, brightness, contrast, exposure, light_compensation):
    """Detect coins in real-time using the camera."""
    width, height = map(int, resolution.split('x'))
    
    # Initialize the camera
    camera = Camera(camera_id, width, height)
    
    # Set camera properties if provided
    if brightness is not None:
        camera.set_brightness(brightness)
    if contrast is not None:
        camera.set_contrast(contrast)
    if exposure is not None:
        camera.set_exposure(exposure)
    
    # Enable light compensation if requested
    if light_compensation:
        camera.enable_light_compensation()
    
    # Load models
    coin_model = CoinModel(model_dir)
    
    # Initialize detector
    detector = CoinDetector(coin_model)
    
    # Load calibration if provided
    if calibration:
        calibration_path = os.path.abspath(calibration)
        if os.path.exists(calibration_path):
            click.echo(f"Using calibration file: {calibration_path}")
            calibration_obj = CameraCalibration()
            if calibration_obj.load(calibration_path):
                detector.calibration = calibration_obj
                if calibration_obj.get_pixels_per_mm():
                    click.echo(f"Calibration loaded: {calibration_obj.get_pixels_per_mm():.2f} pixels per mm")
                else:
                    click.echo("Calibration loaded but no pixels per mm data found")
            else:
                click.echo(f"Failed to load calibration file: {calibration_path}")
        else:
            click.echo(f"Calibration file not found: {calibration_path}")
    
    click.echo("Starting live coin detection. Press 'q' to quit.")
    
    # Start live detection
    cv2.namedWindow('Coin Detection', cv2.WINDOW_NORMAL)
    
    while True:
        # Capture frame
        frame = camera.read_frame()
        if frame is None:
            continue
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Use the same preprocessing as in calibration
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 1000  # Minimum contour area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Find the best coin contour (most circular)
        best_contour = None
        best_circularity = 0
        
        for contour in filtered_contours:
            # Calculate circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Update best contour if this is more circular
            if circularity > best_circularity and circularity > 0.8:  # Good circularity threshold
                best_contour = contour
                best_circularity = circularity
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Draw all contours in green
        cv2.drawContours(display_frame, filtered_contours, -1, (0, 255, 0), 2)
        
        # Process the best contour if found
        if best_contour is not None:
            # Calculate features
            features = detector.analyze_contour(best_contour)
            
            # Match to coin type
            match = detector.match_coin(features)
            
            # Get contour center and diameter
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                position = (cx, cy)
            else:
                position = (0, 0)
                
            # Calculate diameter
            area = cv2.contourArea(best_contour)
            diameter_px = np.sqrt(4 * area / np.pi)
            radius = int(diameter_px / 2)
            
            # Draw the best contour in blue
            cv2.drawContours(display_frame, [best_contour], 0, (255, 0, 0), 2)
            
            # Draw circle around the coin to highlight the perimeter
            cv2.circle(display_frame, position, radius, (0, 0, 255), 2)
            
            # Calculate aspect ratio using fitted ellipse
            if len(best_contour) >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(best_contour)
                # Draw the fitted ellipse
                cv2.ellipse(display_frame, ellipse, (255, 0, 255), 2)
                
                # Get ellipse parameters
                (_, _), (major_axis, minor_axis), _ = ellipse
                current_aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
                
                # Display aspect ratio
                cv2.putText(display_frame, f"Aspect Ratio: {current_aspect_ratio:.2f}", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display circularity
            cv2.putText(display_frame, f"Circularity: {best_circularity:.2f}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display diameter in pixels
            cv2.putText(display_frame, f"Diameter: {diameter_px:.1f}px", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate diameter in mm if calibration is available
            if detector.calibration and detector.calibration.get_pixels_per_mm():
                # If we have aspect ratio information, use it to correct the diameter
                if hasattr(detector.calibration, 'aspect_ratio') and detector.calibration.aspect_ratio != 0 and features['axes'] is not None:
                    major_axis, minor_axis = features['axes']
                    # Correct the minor axis using the calibration aspect ratio
                    corrected_minor = minor_axis * detector.calibration.aspect_ratio
                    # Use geometric mean for a more accurate diameter
                    corrected_diameter_px = np.sqrt(major_axis * corrected_minor)
                    diameter_mm = corrected_diameter_px / detector.calibration.get_pixels_per_mm()
                else:
                    diameter_mm = diameter_px / detector.calibration.get_pixels_per_mm()
                
                # Display diameter in mm
                cv2.putText(display_frame, f"Diameter: {diameter_mm:.1f}mm", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Known diameters of Euro coins in mm
                coin_diameters = {
                    '1_cent': 16.25,
                    '2_cent': 18.75,
                    '5_cent': 21.25,
                    '10_cent': 19.75,
                    '20_cent': 22.25,
                    '50_cent': 24.25,
                    '1_euro': 23.25,
                    '2_euro': 25.75
                }
                
                # Find the closest match based on diameter
                best_match = None
                best_diff = float('inf')
                
                for coin_type, coin_diameter in coin_diameters.items():
                    diff = abs(diameter_mm - coin_diameter)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = coin_type
                
                # Display the identified coin type
                if best_match and best_diff < 2.0:  # Only show if difference is less than 2mm
                    cv2.putText(display_frame, f"Coin: {best_match} (diff: {best_diff:.1f}mm)", 
                               (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Unknown coin", 
                               (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If we have a match from the model, display it for comparison
            if match:
                # Display coin type and confidence from model
                cv2.putText(display_frame, f"Model match: {match['type']} ({match['confidence']:.2f})", 
                           (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Coin Detection', display_frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
    
    # Clean up
    cv2.destroyAllWindows()
    camera.close()

if __name__ == '__main__':
    cli()
