#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models module for Coin Meter
Handles coin model data and training
"""

import os
import cv2
import numpy as np
import json
import logging
from datetime import datetime

try:
    # When running as a module (python -m src.main)
    from src.camera import Camera
except ImportError:
    # When running directly (python coin_meter.py)
    from camera import Camera

class CoinModel:
    """Coin model class for handling training data"""
    
    def __init__(self, model_dir):
        """
        Initialize the model
        
        Args:
            model_dir (str): Directory containing model data
        """
        self.model_dir = model_dir
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing models
        self.load_models()
        
    def load_models(self):
        """Load all model files from the model directory"""
        if not os.path.exists(self.model_dir):
            self.logger.warning(f"Model directory does not exist: {self.model_dir}")
            return
            
        # Get all JSON files in the model directory
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.json')]
        
        for model_file in model_files:
            # Extract coin type from filename
            coin_type = os.path.splitext(model_file)[0]
            
            # Load the model
            model_path = os.path.join(self.model_dir, model_file)
            try:
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                    
                self.models[coin_type] = model_data
                self.logger.info(f"Loaded model for {coin_type}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_path}: {e}")
                
    def get_coin_types(self):
        """
        Get all available coin types
        
        Returns:
            list: List of coin types
        """
        return list(self.models.keys())
        
    def get_coin_data(self, coin_type):
        """
        Get data for a specific coin type
        
        Args:
            coin_type (str): Coin type
            
        Returns:
            dict: Coin data
        """
        return self.models.get(coin_type)
        
    def save(self, coin_type, filename):
        """
        Save model data to a file
        
        Args:
            coin_type (str): Coin type
            filename (str): Output filename
            
        Returns:
            bool: True if saving was successful
        """
        if coin_type not in self.models:
            self.logger.error(f"No model data for {coin_type}")
            return False
            
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.models[coin_type], f, indent=2)
                
            self.logger.info(f"Model saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
            
    def train_from_camera(self, camera, coin_type):
        """
        Train the model using the camera
        
        Args:
            camera (Camera): Camera object
            coin_type (str): Coin type to train
            
        Returns:
            bool: True if training was successful
        """
        self.logger.info(f"Training model for {coin_type}")
        
        # Initialize model data
        model_data = {
            'coin_type': coin_type,
            'contour_points': [],
            'hu_moments': [],
            'circularity': 0,
            'aspect_ratio': 0,
            'equivalent_diameter': 0,
            'radius_mm': self._get_coin_radius_mm(coin_type),
            'training_samples': 0,
            'training_time': datetime.now().isoformat()
        }
        
        # Start interactive training
        cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
        
        samples = []
        while True:
            # Capture frame
            frame = camera.read_frame()
            if frame is None:
                continue
                
            # Process the frame
            processed, contour = self._process_training_frame(frame)
            
            # Display the frame
            display_frame = frame.copy()
            if contour is not None:
                # Draw the contour
                cv2.drawContours(display_frame, [contour], 0, (0, 255, 0), 2)
                
                # Calculate features
                features = self._calculate_features(contour)
                
                # Display features
                cv2.putText(display_frame, f"Circularity: {features['circularity']:.2f}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Aspect Ratio: {features['aspect_ratio']:.2f}", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show instructions
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to finish", 
                           (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Show instructions
                cv2.putText(display_frame, "No coin detected", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'q' to finish", 
                           (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Show the frame
            cv2.imshow('Training', display_frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' to quit
            if key == ord('q'):
                break
                
            # 'c' to capture
            elif key == ord('c') and contour is not None:
                # Calculate features
                features = self._calculate_features(contour)
                
                # Add to samples
                samples.append(features)
                self.logger.info(f"Captured sample {len(samples)}")
                
        # Clean up
        cv2.destroyAllWindows()
        
        # Process samples
        if len(samples) > 0:
            # Calculate average features
            model_data['circularity'] = np.mean([s['circularity'] for s in samples])
            model_data['aspect_ratio'] = np.mean([s['aspect_ratio'] for s in samples])
            model_data['equivalent_diameter'] = np.mean([s['equivalent_diameter'] for s in samples])
            
            # Use the best sample for contour points and Hu moments
            best_sample = max(samples, key=lambda s: s['circularity'])
            model_data['contour_points'] = best_sample['contour_points']
            model_data['hu_moments'] = best_sample['hu_moments']
            
            # Update training info
            model_data['training_samples'] = len(samples)
            model_data['training_time'] = datetime.now().isoformat()
            
            # Save to model
            self.models[coin_type] = model_data
            
            self.logger.info(f"Model trained with {len(samples)} samples")
            return True
        else:
            self.logger.warning("No samples collected")
            return False
            
    def _process_training_frame(self, frame):
        """
        Process a frame for training
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (processed_frame, contour)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Filter by area and circularity
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if area > 1000 and circularity > 0.7:  # Minimum area and circularity
                return binary, largest_contour
                
        return binary, None
        
    def _calculate_features(self, contour):
        """
        Calculate features for a contour
        
        Args:
            contour (numpy.ndarray): Contour
            
        Returns:
            dict: Features
        """
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Fit an ellipse
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Calculate aspect ratio
            major_axis = max(axes)
            minor_axis = min(axes)
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
        else:
            aspect_ratio = 1
            
        # Calculate equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        # Calculate Hu moments
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten().tolist()
        
        # Simplify contour for storage
        epsilon = 0.01 * perimeter
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert contour to list of points
        contour_points = approx_contour.reshape(-1, 2).tolist()
        
        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'equivalent_diameter': equivalent_diameter,
            'hu_moments': hu_moments,
            'contour_points': contour_points
        }
        
    def _get_coin_radius_mm(self, coin_type):
        """
        Get the radius of a coin in mm
        
        Args:
            coin_type (str): Coin type
            
        Returns:
            float: Radius in mm
        """
        # Euro coin radii in mm
        radii = {
            '1_cent': 8.0,
            '2_cent': 9.5,
            '5_cent': 10.75,
            '10_cent': 9.75,
            '20_cent': 11.25,
            '50_cent': 12.0,
            '1_euro': 11.75,
            '2_euro': 12.5
        }
        
        return radii.get(coin_type, 0)
