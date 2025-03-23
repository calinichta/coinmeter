#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check if required modules can be imported
"""

import sys

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {module_name} is available")
        return True
    except ImportError as e:
        print(f"✗ {module_name} is NOT available: {e}")
        return False

if __name__ == "__main__":
    print("Checking required modules...")
    
    # Check required modules
    modules = [
        "cv2",  # OpenCV
        "numpy",
        "click",
        "json",
        "logging",
        "datetime",
        "os",
        "sys",
        "unittest"
    ]
    
    success = True
    for module in modules:
        if not check_import(module):
            success = False
    
    if success:
        print("\nAll required modules are available!")
    else:
        print("\nSome required modules are missing. Please install them using:")
        print("pip install -r requirements.txt")
    
    print("\nPython version:", sys.version)
