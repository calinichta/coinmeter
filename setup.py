#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="coin_meter",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A CLI-based Euro coin recognition program using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coin_meter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "coin_meter=src.main:cli",
        ],
    },
)
