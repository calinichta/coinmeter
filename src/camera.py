#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera module for Coin Meter
Handles camera operations, calibration, and image capture
"""

import os
import cv2
import numpy as np
import json
import logging
from datetime import datetime

try:
    # When running as a module (python -m src.main)
    from src.utils import ensure_directories
except ImportError:
    # When running directly (python coin_meter.py)
    from utils import ensure_directories

class Camera:
    """Camera class for handling webcam operations"""
    
    def __init__(self, camera_id=0, width=1280, height=720):
        """
        Initialize the camera
        
        Args:
            camera_id (int): Camera ID to use
            width (int): Camera width resolution
            height (int): Camera height resolution
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.light_compensation = False
        self.logger = logging.getLogger(__name__)
        
    def open(self):
        """Open the camera"""
        if self.cap is not None and self.cap.isOpened():
            return True
            
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open camera {self.camera_id}")
            return False
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.logger.info(f"Camera {self.camera_id} opened with resolution {self.width}x{self.height}")
        return True
        
    def close(self):
        """Close the camera"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.logger.info("Camera closed")
            
    def __del__(self):
        """Destructor to ensure camera is closed"""
        self.close()
        
    def set_brightness(self, value):
        """Set camera brightness"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
            self.logger.info(f"Camera brightness set to {value}")
            
    def set_contrast(self, value):
        """Set camera contrast"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_CONTRAST, value)
            self.logger.info(f"Camera contrast set to {value}")
            
    def set_exposure(self, value):
        """Set camera exposure"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            self.logger.info(f"Camera exposure set to {value}")
            
    def enable_light_compensation(self, enable=True):
        """Enable/disable automatic histogram equalization"""
        self.light_compensation = enable
        self.logger.info(f"Light compensation {'enabled' if enable else 'disabled'}")
        
    def read_frame(self):
        """Read a frame from the camera"""
        if not self.open():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to read frame from camera")
            return None
            
        # Apply light compensation if enabled
        if self.light_compensation:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        return frame
        
    def capture(self, filename=None):
        """
        Capture a single frame and optionally save to file
        
        Args:
            filename (str): Filename to save the image to
            
        Returns:
            numpy.ndarray: The captured frame
        """
        frame = self.read_frame()
        
        if frame is not None and filename:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Image saved to {filename}")
            
        return frame
        
    def capture_interactive(self, filename_template=None):
        """
        Interactive capture mode with live preview
        
        Args:
            filename_template (str): Template for filenames (will append timestamp)
        """
        if not self.open():
            return
            
        self.logger.info("Starting interactive capture mode")
        self.logger.info("Press 'c' to capture, 'q' to quit")
        
        while True:
            frame = self.read_frame()
            if frame is None:
                break
                
            # Display the frame
            cv2.imshow('Coin Meter - Capture', frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' to quit
            if key == ord('q'):
                break
                
            # 'c' to capture
            elif key == ord('c'):
                if filename_template:
                    # Generate filename with timestamp if not provided
                    if '{timestamp}' in filename_template:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = filename_template.format(timestamp=timestamp)
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base, ext = os.path.splitext(filename_template)
                        filename = f"{base}_{timestamp}{ext}"
                        
                    # Save the frame
                    cv2.imwrite(filename, frame)
                    self.logger.info(f"Image captured and saved to {filename}")
                else:
                    self.logger.info("Image captured (not saved)")
                    
        # Clean up
        cv2.destroyAllWindows()
        self.close()


class CameraCalibration:
    """Camera calibration using chessboard pattern or coin reference"""
    
    def __init__(self):
        """Initialize the calibration"""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.pixels_per_mm = None
        self.reference_coin = None
        self.reference_diameter_mm = None
        self.aspect_ratio = 1.0  # Default aspect ratio (1.0 means perfect circle)
        self.calibrated = False
        self.logger = logging.getLogger(__name__)
        
        # Chessboard parameters (9x6 internal corners)
        self.chessboard_size = (9, 6)
        self.square_size = 1.0  # Size of a square in the chessboard (arbitrary unit)
        
    def calibrate(self, camera, num_samples=10):
        """
        Calibrate the camera using a chessboard pattern
        
        Args:
            camera (Camera): Camera object to calibrate
            num_samples (int): Number of samples to collect
            
        Returns:
            bool: True if calibration was successful
        """
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        self.logger.info("Starting camera calibration")
        self.logger.info(f"Please show a {self.chessboard_size[0]}x{self.chessboard_size[1]} chessboard pattern")
        self.logger.info(f"Need {num_samples} good samples")
        
        # Open a window for preview
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        
        samples_collected = 0
        while samples_collected < num_samples:
            frame = camera.read_frame()
            if frame is None:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            # Draw the corners
            display_frame = frame.copy()
            if ret:
                # Refine the corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw the corners
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                
                # Display status
                cv2.putText(display_frame, f"Sample {samples_collected+1}/{num_samples}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Calibration', display_frame)
                key = cv2.waitKey(500) & 0xFF
                
                # 'q' to quit
                if key == ord('q'):
                    break
                    
                # Collect the sample
                objpoints.append(objp)
                imgpoints.append(corners)
                samples_collected += 1
                self.logger.info(f"Collected sample {samples_collected}/{num_samples}")
            else:
                # Display status
                cv2.putText(display_frame, "No chessboard detected", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Calibration', display_frame)
                key = cv2.waitKey(100) & 0xFF
                
                # 'q' to quit
                if key == ord('q'):
                    break
        
        # Clean up
        cv2.destroyAllWindows()
        
        # Perform calibration if we have enough samples
        if samples_collected >= 3:  # Need at least 3 samples for calibration
            self.logger.info("Calculating calibration parameters...")
            
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)
                
            if ret:
                self.calibrated = True
                self.logger.info("Calibration successful")
                return True
            else:
                self.logger.error("Calibration failed")
                return False
        else:
            self.logger.error(f"Not enough samples collected ({samples_collected})")
            return False
            
    def undistort(self, image):
        """
        Undistort an image using the calibration parameters
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Undistorted image
        """
        if not self.calibrated:
            self.logger.warning("Camera not calibrated, returning original image")
            return image
            
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        
    def calibrate_with_coin(self, camera, coin_type='2_euro', num_samples=5):
        """
        Calibrate the camera using a coin as reference
        
        Args:
            camera (Camera): Camera object to calibrate
            coin_type (str): Coin type to use as reference (default: 2_euro)
            num_samples (int): Number of samples to collect
            
        Returns:
            bool: True if calibration was successful
        """
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
        
        # Get the diameter for the selected coin type
        if coin_type in coin_diameters:
            coin_diameter_mm = coin_diameters[coin_type]
        else:
            self.logger.warning(f"Unknown coin type: {coin_type}, using 2_euro as default")
            coin_diameter_mm = coin_diameters['2_euro']
        
        # Collect samples
        diameters_px = []
        aspect_ratios = []
        
        # Start interactive calibration
        cv2.namedWindow('Coin Calibration', cv2.WINDOW_NORMAL)
        
        samples_collected = 0
        while samples_collected < num_samples:
            # Capture frame
            frame = camera.read_frame()
            if frame is None:
                continue
                
            # Preprocess image and find contours
            processed = self._preprocess_image(frame)
            contours = self._find_coin_contours(processed)
            
            # Find best coin contour (most circular)
            best_contour, circularity = self._find_best_coin_contour(contours)
            
            # Display
            display_frame = frame.copy()
            
            # Add text guide
            cv2.putText(display_frame, "Place a coin anywhere in the frame", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if best_contour is not None and circularity > 0.5:  # Good circularity
                # Calculate diameter and aspect ratio
                area = cv2.contourArea(best_contour)
                diameter_px = np.sqrt(4 * area / np.pi)
                
                # Get contour center
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                
                # Calculate aspect ratio using fitted ellipse
                ellipse = cv2.fitEllipse(best_contour) if len(best_contour) >= 5 else None
                
                if ellipse is not None:
                    (_, _), (major_axis, minor_axis), _ = ellipse
                    current_aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
                    
                    # Draw the fitted ellipse
                    cv2.ellipse(display_frame, ellipse, (255, 0, 255), 2)
                else:
                    current_aspect_ratio = 1.0
                
                # Draw contour with fixed color
                cv2.drawContours(display_frame, [best_contour], 0, (0, 255, 0), 2)
                
                # Draw circle around the coin to highlight the perimeter
                if M["m00"] != 0:
                    radius = int(diameter_px / 2)
                    cv2.circle(display_frame, (cx, cy), radius, (0, 0, 255), 2)
                
                # Display info
                cv2.putText(display_frame, f"Diameter: {diameter_px:.1f}px", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Circularity: {circularity:.2f}", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Aspect Ratio: {current_aspect_ratio:.2f}", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Sample {samples_collected+1}/{num_samples}", 
                           (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show instructions
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                           (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Coin Calibration', display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                # 'q' to quit
                if key == ord('q'):
                    break
                    
                # 'c' to capture
                elif key == ord('c'):
                    diameters_px.append(diameter_px)
                    if ellipse is not None:
                        aspect_ratios.append(current_aspect_ratio)
                    samples_collected += 1
                    self.logger.info(f"Collected sample {samples_collected}/{num_samples}")
            else:
                # Show instructions
                cv2.putText(display_frame, "No coin detected or poor circularity", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Place a {coin_type} coin anywhere in view", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'q' to quit", 
                           (20, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Coin Calibration', display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                # 'q' to quit
                if key == ord('q'):
                    break
        
        # Clean up
        cv2.destroyAllWindows()
        
        # Calculate calibration if we have enough samples
        if samples_collected > 0:
            # Calculate average diameter
            avg_diameter_px = np.mean(diameters_px)
            
            # Calculate pixels per mm
            pixels_per_mm = avg_diameter_px / coin_diameter_mm
            
            # Calculate average aspect ratio
            avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1.0
            
            # Store calibration data
            self.pixels_per_mm = pixels_per_mm
            self.reference_coin = coin_type
            self.reference_diameter_mm = coin_diameter_mm
            self.aspect_ratio = avg_aspect_ratio
            self.calibrated = True
            
            self.logger.info(f"Calibration successful: {pixels_per_mm:.2f} pixels per mm, aspect ratio: {avg_aspect_ratio:.2f}")
            return True
        else:
            self.logger.error("Not enough samples collected")
            return False
            
    def _preprocess_image(self, image):
        """
        Preprocess image for coin detection
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return opening
        
    def _find_coin_contours(self, preprocessed):
        """
        Find contours in preprocessed image
        
        Args:
            preprocessed (numpy.ndarray): Preprocessed image
            
        Returns:
            list: List of contours
        """
        # Find contours
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 1000  # Minimum contour area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return filtered_contours
        
    def _find_best_coin_contour(self, contours):
        """
        Find the contour that most resembles a coin
        
        Args:
            contours (list): List of contours
            
        Returns:
            tuple: (best_contour, circularity)
        """
        if not contours:
            return None, 0
            
        best_contour = None
        best_circularity = 0
        
        for contour in contours:
            # Calculate circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Update best contour if this is more circular
            if circularity > best_circularity:
                best_contour = contour
                best_circularity = circularity
                
        return best_contour, best_circularity
    
    def save(self, filename):
        """
        Save calibration parameters to a file
        
        Args:
            filename (str): Output filename
        """
        if not self.calibrated:
            self.logger.error("Cannot save calibration: camera not calibrated")
            return False
            
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        data = {
            'calibration_time': datetime.now().isoformat()
        }
        
        # Add camera matrix and distortion coefficients if available
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            data['camera_matrix'] = self.camera_matrix.tolist()
            data['dist_coeffs'] = self.dist_coeffs.tolist()
            
        # Add pixels per mm if available
        if self.pixels_per_mm is not None:
            data['pixels_per_mm'] = self.pixels_per_mm
            data['reference_coin'] = self.reference_coin
            data['reference_diameter_mm'] = self.reference_diameter_mm
            data['aspect_ratio'] = self.aspect_ratio
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Calibration saved to {filename}")
        return True
        
    def load(self, filename):
        """
        Load calibration parameters from a file
        
        Args:
            filename (str): Input filename
            
        Returns:
            bool: True if loading was successful
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load camera matrix and distortion coefficients if available
            if 'camera_matrix' in data and 'dist_coeffs' in data:
                self.camera_matrix = np.array(data['camera_matrix'])
                self.dist_coeffs = np.array(data['dist_coeffs'])
                
            # Load pixels per mm if available
            if 'pixels_per_mm' in data:
                self.pixels_per_mm = data['pixels_per_mm']
                self.reference_coin = data.get('reference_coin')
                self.reference_diameter_mm = data.get('reference_diameter_mm')
                self.aspect_ratio = data.get('aspect_ratio', 1.0)
                
            self.calibrated = True
            
            self.logger.info(f"Calibration loaded from {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False
            
    def get_pixels_per_mm(self):
        """
        Get the pixels per mm ratio
        
        Returns:
            float: Pixels per mm or None if not calibrated
        """
        return self.pixels_per_mm if self.calibrated else None
