#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coin Meter - CLI-based coin recognition program
Wrapper script to run the program from the command line
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main CLI function
from src.main import cli

if __name__ == '__main__':
    # Run the CLI with the program name
    cli(prog_name='coin_meter.py')
