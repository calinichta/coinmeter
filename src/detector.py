#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detector module for Coin Meter
Handles coin detection using contour analysis
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime

try:
    # When running as a module (python -m src.main)
    from src.models import CoinModel
except ImportError:
    # When running directly (python coin_meter.py)
    from models import CoinModel

class CoinDetector:
    """Coin detector using contour analysis"""
    
    def __init__(self, coin_model, calibration=None):
        """
        Initialize the detector
        
        Args:
            coin_model (CoinModel): Model containing coin reference data
            calibration (CameraCalibration, optional): Camera calibration data
        """
        self.coin_model = coin_model
        self.calibration = calibration
        self.logger = logging.getLogger(__name__)
        
    def preprocess_image(self, image):
        """
        Preprocess the image for contour detection
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(blurred)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return opening
        
    def find_contours(self, preprocessed):
        """
        Find contours in the preprocessed image
        
        Args:
            preprocessed (numpy.ndarray): Preprocessed image
            
        Returns:
            list: List of contours
        """
        # Find contours
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove small noise
        min_area = 1000  # Minimum contour area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        self.logger.info(f"Found {len(filtered_contours)} potential coins")
        return filtered_contours
        
    def analyze_contour(self, contour):
        """
        Analyze a contour to extract features
        
        Args:
            contour (numpy.ndarray): Contour to analyze
            
        Returns:
            dict: Contour features
        """
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Calculate contour perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity (4*pi*area/perimeter^2)
        # A perfect circle has circularity = 1
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Fit an ellipse to the contour
        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Calculate aspect ratio (major axis / minor axis)
            major_axis = max(axes)
            minor_axis = min(axes)
            raw_aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
            
            # Correct aspect ratio using calibration if available
            if self.calibration and hasattr(self.calibration, 'aspect_ratio') and self.calibration.aspect_ratio != 0:
                # Normalize the aspect ratio by dividing by the camera's aspect ratio
                aspect_ratio = raw_aspect_ratio / self.calibration.aspect_ratio
            else:
                aspect_ratio = raw_aspect_ratio
            
            # Calculate equivalent diameter
            equivalent_diameter = np.sqrt(4 * area / np.pi)
        else:
            center = None
            axes = None
            angle = None
            aspect_ratio = 1
            equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        # Return features
        return {
            'contour': contour,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'center': center,
            'axes': axes,
            'angle': angle,
            'aspect_ratio': aspect_ratio,
            'equivalent_diameter': equivalent_diameter
        }
        
    def match_coin(self, contour_features):
        """
        Match a contour to a coin type
        
        Args:
            contour_features (dict): Contour features
            
        Returns:
            dict: Matched coin with type and confidence
        """
        # Get all coin types from the model
        coin_types = self.coin_model.get_coin_types()
        
        best_match = None
        best_confidence = 0
        
        for coin_type in coin_types:
            # Get reference data for this coin type
            reference = self.coin_model.get_coin_data(coin_type)
            
            # Skip if no reference data
            if not reference:
                continue
                
            # Calculate match confidence based on features
            confidence = self._calculate_match_confidence(contour_features, reference)
            
            # Update best match if this is better
            if confidence > best_confidence:
                best_match = coin_type
                best_confidence = confidence
                
        # Return the best match
        if best_match and best_confidence > 0.5:  # Minimum confidence threshold
            return {
                'type': best_match,
                'confidence': best_confidence
            }
        else:
            return None
            
    def _calculate_match_confidence(self, contour_features, reference):
        """
        Calculate match confidence between contour and reference
        
        Args:
            contour_features (dict): Contour features
            reference (dict): Reference data for a coin type
            
        Returns:
            float: Match confidence (0-1)
        """
        # Extract features
        contour = contour_features['contour']
        
        # Calculate shape matching using Hu moments
        contour_hu = cv2.HuMoments(cv2.moments(contour)).flatten()
        reference_hu = np.array(reference.get('hu_moments', [0] * 7))
        
        # Calculate match using shape context distance
        if 'contour_points' in reference and len(reference['contour_points']) >= 5:
            reference_contour = np.array(reference['contour_points'], dtype=np.float32).reshape(-1, 1, 2)
            
            # Normalize contours to same size
            ref_center, ref_axes, _ = cv2.fitEllipse(reference_contour)
            ref_max_axis = max(ref_axes)
            
            if contour_features['axes'] is not None:
                contour_max_axis = max(contour_features['axes'])
                scale = ref_max_axis / contour_max_axis if contour_max_axis > 0 else 1
                
                # Scale contour to match reference size
                M = np.array([[scale, 0], [0, scale]])
                scaled_contour = np.array([np.dot(M, point.flatten()) for point in contour]).reshape(-1, 1, 2)
                
                # Calculate shape matching
                try:
                    # Use contour matching
                    # Convert contours to single-channel format if needed
                    if len(reference_contour.shape) > 3:
                        reference_contour_2d = reference_contour.reshape(-1, 2).astype(np.float32)
                    else:
                        reference_contour_2d = reference_contour
                        
                    if len(scaled_contour.shape) > 3:
                        scaled_contour_2d = scaled_contour.reshape(-1, 2).astype(np.float32)
                    else:
                        scaled_contour_2d = scaled_contour
                    
                    # Create grayscale images with contours
                    ref_img = np.zeros((500, 500), dtype=np.uint8)
                    cv2.drawContours(ref_img, [reference_contour_2d.astype(np.int32)], 0, 255, 1)
                    
                    test_img = np.zeros((500, 500), dtype=np.uint8)
                    cv2.drawContours(test_img, [scaled_contour_2d.astype(np.int32)], 0, 255, 1)
                    
                    # Use matchShapes on the images
                    shape_match = cv2.matchShapes(ref_img, test_img, cv2.CONTOURS_MATCH_I3, 0.0)
                    shape_match = 1.0 / (1.0 + shape_match)
                except Exception as e:
                    # If matchShapes fails, use a simpler approach
                    self.logger.warning(f"Shape matching failed: {e}")
                    shape_match = 0.5  # Default value
            else:
                shape_match = 0.5  # Default if we can't calculate
        else:
            # Fallback to Hu moments matching
            shape_match = 0
            for i in range(min(len(contour_hu), len(reference_hu))):
                if contour_hu[i] != 0 and reference_hu[i] != 0:
                    shape_match += 1.0 / (1.0 + abs(contour_hu[i] - reference_hu[i]))
            shape_match /= min(len(contour_hu), len(reference_hu))
        
        # Calculate circularity match
        circularity_match = 1.0 - abs(contour_features['circularity'] - reference.get('circularity', 1.0))
        
        # Calculate aspect ratio match
        aspect_ratio_match = 1.0 - abs(contour_features['aspect_ratio'] - reference.get('aspect_ratio', 1.0))
        
        # Calculate diameter match if reference has radius
        if 'radius_mm' in reference and reference['radius_mm'] > 0:
            # If we have calibration with pixels_per_mm, use it for accurate size comparison
            if self.calibration and self.calibration.get_pixels_per_mm():
                # Convert contour diameter from pixels to mm, accounting for aspect ratio
                if hasattr(self.calibration, 'aspect_ratio') and self.calibration.aspect_ratio != 0:
                    # Calculate the geometric mean of the axes to get a corrected diameter
                    if contour_features['axes'] is not None:
                        major_axis, minor_axis = contour_features['axes']
                        # Correct the minor axis using the calibration aspect ratio
                        corrected_minor = minor_axis * self.calibration.aspect_ratio
                        # Use geometric mean for a more accurate diameter
                        corrected_diameter_px = np.sqrt(major_axis * corrected_minor)
                    else:
                        corrected_diameter_px = contour_features['equivalent_diameter']
                    
                    contour_diameter_mm = corrected_diameter_px / self.calibration.get_pixels_per_mm()
                else:
                    contour_diameter_mm = contour_features['equivalent_diameter'] / self.calibration.get_pixels_per_mm()
                
                # Calculate expected diameter in mm (2 * radius)
                expected_diameter_mm = 2 * reference['radius_mm']
                # Calculate match based on actual mm measurements
                diameter_match = 1.0 - min(abs(contour_diameter_mm - expected_diameter_mm) / expected_diameter_mm, 1.0)
            else:
                # Fallback to relative size comparison
                diameter_match = 1.0 - abs(
                    contour_features['equivalent_diameter'] / reference.get('equivalent_diameter', 1.0) - 1.0)
        else:
            diameter_match = 1.0
            
        # Combine matches with weights
        weights = {
            'shape': 0.5,
            'circularity': 0.2,
            'aspect_ratio': 0.2,
            'diameter': 0.1
        }
        
        confidence = (
            weights['shape'] * shape_match +
            weights['circularity'] * circularity_match +
            weights['aspect_ratio'] * aspect_ratio_match +
            weights['diameter'] * diameter_match
        )
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
        
    def detect(self, image_path, save_debug=False, calibration_file=None):
        """
        Detect coins in an image
        
        Args:
            image_path (str): Path to the image
            save_debug (bool): Whether to save debug images
            calibration_file (str, optional): Path to calibration file
            
        Returns:
            list: List of detected coins
        """
        # Load calibration if provided and not already loaded
        if calibration_file and not self.calibration:
            try:
                from src.camera import CameraCalibration
            except ImportError:
                from camera import CameraCalibration
                
            self.calibration = CameraCalibration()
            self.calibration.load(calibration_file)
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return []
            
        # Preprocess the image
        preprocessed = self.preprocess_image(image)
        
        # Find contours
        contours = self.find_contours(preprocessed)
        
        # Analyze contours and match coins
        detected_coins = []
        for contour in contours:
            # Analyze contour
            features = self.analyze_contour(contour)
            
            # Match to coin type
            match = self.match_coin(features)
            
            # Add to results if matched
            if match:
                # Create coin data
                coin_data = {
                    'type': match['type'],
                    'confidence': match['confidence'],
                    'position': features['center'] if features['center'] is not None else (0, 0),
                    'diameter_px': features['equivalent_diameter']
                }
                
                # Add diameter in mm if calibration is available
                if self.calibration and self.calibration.get_pixels_per_mm():
                    # If we have aspect ratio information, use it to correct the diameter
                    if hasattr(self.calibration, 'aspect_ratio') and self.calibration.aspect_ratio != 0 and features['axes'] is not None:
                        major_axis, minor_axis = features['axes']
                        # Correct the minor axis using the calibration aspect ratio
                        corrected_minor = minor_axis * self.calibration.aspect_ratio
                        # Use geometric mean for a more accurate diameter
                        corrected_diameter_px = np.sqrt(major_axis * corrected_minor)
                        coin_data['diameter_mm'] = corrected_diameter_px / self.calibration.get_pixels_per_mm()
                    else:
                        coin_data['diameter_mm'] = features['equivalent_diameter'] / self.calibration.get_pixels_per_mm()
                    
                detected_coins.append(coin_data)
                
        # Save debug image if requested
        if save_debug:
            debug_image = image.copy()
            
            # Draw contours
            cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
            
            # Draw detected coins
            for coin in detected_coins:
                # Draw circle at coin position
                position = coin['position']
                radius = int(coin['diameter_px'] / 2)
                cv2.circle(debug_image, (int(position[0]), int(position[1])), radius, (0, 0, 255), 2)
                
                # Draw coin type, confidence, and size
                if 'diameter_mm' in coin:
                    text = f"{coin['type']}: {coin['confidence']:.2f} ({coin['diameter_mm']:.1f}mm)"
                else:
                    text = f"{coin['type']}: {coin['confidence']:.2f}"
                cv2.putText(debug_image, text, (int(position[0]), int(position[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                           
            # Save debug image
            base_path, ext = os.path.splitext(image_path)
            debug_path = f"{base_path}_debug{ext}"
            cv2.imwrite(debug_path, debug_image)
            self.logger.info(f"Debug image saved to {debug_path}")
            
        return detected_coins
