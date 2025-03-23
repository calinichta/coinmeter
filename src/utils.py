#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities module for Coin Meter
Contains helper functions
"""

import os
import sys
import logging
import cv2
from datetime import datetime

def setup_logging(verbose=False):
    """
    Setup logging configuration
    
    Args:
        verbose (bool): Whether to enable verbose logging
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"coin_meter_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized (verbose={verbose})")
    logging.info(f"Log file: {log_file}")

def ensure_directories(directories):
    """
    Ensure that directories exist, creating them if necessary
    
    Args:
        directories (list): List of directory paths
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.debug(f"Ensured directory exists: {directory}")

def get_timestamp():
    """
    Get a formatted timestamp
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_coin_value(coin_type):
    """
    Format a coin type as a human-readable value
    
    Args:
        coin_type (str): Coin type (e.g., '1_cent', '2_euro')
        
    Returns:
        str: Formatted value (e.g., '1 Cent', '2 Euro')
    """
    if not coin_type:
        return "Unknown"
        
    parts = coin_type.split('_')
    if len(parts) != 2:
        return coin_type
        
    value, unit = parts
    
    # Capitalize unit
    if unit == 'cent':
        unit = 'Cent'
    elif unit == 'euro':
        unit = 'Euro'
        
    return f"{value} {unit}"

def calculate_total_value(coins):
    """
    Calculate the total value of a list of coins in euros
    
    Args:
        coins (list): List of coin dictionaries with 'type' key
        
    Returns:
        float: Total value in euros
    """
    # Coin values in euros
    values = {
        '1_cent': 0.01,
        '2_cent': 0.02,
        '5_cent': 0.05,
        '10_cent': 0.10,
        '20_cent': 0.20,
        '50_cent': 0.50,
        '1_euro': 1.0,
        '2_euro': 2.0
    }
    
    total = 0.0
    for coin in coins:
        coin_type = coin.get('type')
        if coin_type in values:
            total += values[coin_type]
            
    return total

def list_available_cameras():
    """
    List all available cameras
    
    Returns:
        list: List of tuples (camera_id, camera_name)
    """
    available_cameras = []
    
    # Try to open cameras starting from index 0
    for i in range(10):  # Check up to 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera name if possible
            camera_name = f"Camera {i}"
            try:
                # Try to get camera name (not supported on all platforms)
                camera_name = cap.getBackendName()
            except:
                pass
                
            available_cameras.append((i, camera_name))
            cap.release()
        else:
            # If we can't open this camera, we've probably reached the end
            # But continue checking a few more indices just in case
            if i >= 3 and len(available_cameras) == 0:
                break
    
    return available_cameras

def generate_summary(coins):
    """
    Generate a summary of detected coins
    
    Args:
        coins (list): List of coin dictionaries with 'type' key
        
    Returns:
        dict: Summary with counts and total value
    """
    # Count coins by type
    counts = {}
    for coin in coins:
        coin_type = coin.get('type')
        if coin_type:
            counts[coin_type] = counts.get(coin_type, 0) + 1
            
    # Calculate total value
    total_value = calculate_total_value(coins)
    
    # Format summary
    summary = {
        'total_coins': len(coins),
        'coin_counts': counts,
        'total_value_eur': total_value,
        'formatted_value': f"{total_value:.2f} €"
    }
    
    return summary
