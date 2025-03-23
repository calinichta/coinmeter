#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the main.py file with the --help flag
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    # Get the path to the main.py file
    main_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "main.py")
    
    # Check if the file exists
    if not os.path.exists(main_py):
        print(f"Error: {main_py} does not exist")
        sys.exit(1)
    
    # Run the command
    print(f"Running: python {main_py} --help")
    try:
        result = subprocess.run(["python", main_py, "--help"], 
                               capture_output=True, text=True, check=True)
        print("\nOutput:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("\nError:")
        print(e.stderr)
