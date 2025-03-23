# Coin Meter

A CLI-based Euro coin recognition program using computer vision.

## Overview

Coin Meter is a Python-based CLI application that uses a webcam to recognize Euro coins based on their shape (contour analysis). The software is designed to run on Windows, Linux, and Raspberry Pi.

## Features

- **Camera Calibration**: 
  - Automatic or semi-automatic calibration to correct lens distortion using a chessboard pattern
  - Alternative calibration using a 2 Euro coin as a reference for accurate size measurements
  - Visual overlay guide for precise coin positioning during calibration
- **Image Capture**: Manual or automatic image capture with live preview
- **Coin Recognition**: Recognizes Euro coins (1, 2, 5, 10, 20, 50 cents, 1 and 2 Euro) based on contour analysis
- **Size Measurement**: Accurate size measurements in millimeters when using calibration
- **Training Mode**: Create custom training data for each coin type
- **Debug Visualization**: Option to save debug images with detected contours

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Click

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/coin_meter.git
   cd coin_meter
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Coin Meter provides several modes of operation. You can run the program using either the wrapper script or the module directly:

```
# Using the wrapper script
python coin_meter.py [command] [options]

# Using the module directly
python -m src.main [command] [options]
```

### List Available Cameras

List all available cameras:

```
python coin_meter.py list_cameras
```

### Camera Calibration

Calibrate the camera using a chessboard pattern:

```
python coin_meter.py calibrate --camera-id 0 --resolution 1280x720
```

Or calibrate using a 2 Euro coin as a reference:

```
python coin_meter.py calibrate-with-coin --camera-id 0 --resolution 1280x720
```

### Image Capture

Capture images from the camera:

```
python coin_meter.py capture --camera-id 0 --resolution 1280x720 --brightness 50 --contrast 50
```

### Coin Detection

Detect coins in an image or directory of images:

```
python coin_meter.py detect --input ./pictures/image.jpg --model-dir ./models --save-debug
```

Use a calibration file for accurate size measurements:

```
python coin_meter.py detect --input ./pictures/image.jpg --calibration ./calibration/coin_calibration.json --save-debug
```

### Live Coin Detection

Detect coins in real-time using the camera:

```
python coin_meter.py detect-live --camera-id 1
```

Use a calibration file for accurate size measurements in real-time:

```
python coin_meter.py detect-live --camera-id 1 --calibration ./calibration/coin_calibration.json
```

### Training

Train the model for a specific coin type:

```
python coin_meter.py train-from-camera --camera-id 0 --resolution 1280x720 --coin-type 1_euro
```

## Command Line Options

| Argument             | Description |
|----------------------|-------------|
| `--mode`             | Operation mode: `"list_cameras"`, `"calibrate"`, `"calibrate-with-coin"`, `"capture"`, `"detect"`, `"detect-live"`, `"train-from-camera"` |
| `--camera-id`        | Camera index (e.g., `0`) |
| `--resolution`       | Camera resolution (e.g., `1280x720`) |
| `--brightness`       | Camera brightness setting |
| `--contrast`         | Camera contrast setting |
| `--exposure`         | Camera exposure setting |
| `--light-compensation` | Enable automatic histogram equalization |
| `--input`            | Input image or directory path for analysis |
| `--model-dir`        | Directory containing training data |
| `--output`           | Output file for results |
| `--save-debug`       | Save debug images with contours |
| `--verbose`          | Enable verbose logging |

## Project Structure

```
coin_meter/
├── calibration/       # Camera calibration data
├── logs/              # Log files
├── models/            # Coin model data
├── pictures/          # Captured images
└── src/               # Source code
    ├── main.py        # CLI entry point
    ├── camera.py      # Camera handling
    ├── detector.py    # Coin detection
    ├── models.py      # Model management
    └── utils.py       # Utility functions
```

## Lighting Conditions

For optimal results:
- Use even, diffuse lighting
- Avoid reflections and shadows
- Place coins on a contrasting background

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"# coinmeter" 
