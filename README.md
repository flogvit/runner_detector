# Runner Detector and Bib Number OCR

This script uses a YOLO model for person detection and EasyOCR for bib number recognition. It crops detected persons from input images while ensuring a specified minimum height and aspect ratio. The cropped images are saved with filenames that include the detected bib number.

## Features

- **Detection & OCR:** Uses YOLOv8 for person detection and EasyOCR for reading bib numbers.
- **Customizable Output:** Change the input/output directories, minimum crop height, and desired aspect ratio via command-line arguments.
- **Batch Processing:** Automatically processes images in batches based on your hardware capabilities.
- **Concurrency:** Utilizes multithreading to process OCR on multiple detections concurrently.

## Requirements

- Python 3.8 or later
- [OpenCV](https://pypi.org/project/opencv-python/)
- [PyTorch](https://pytorch.org/) (installation instructions [here](https://pytorch.org/get-started/locally/))
- [Ultralytics YOLO](https://pypi.org/project/ultralytics/)
- [EasyOCR](https://pypi.org/project/easyocr/)

## Installation

1. **Clone this repository or download the script.**

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install the required packages:
    ```bash
   pip install -r requirements.txt
    ```
   Note: For PyTorch, please follow the official installation guide to select the correct command for your system and CUDA version.

## Usage
Place your input images in a directory (default is in). Then run the script using:
```bash
# Standard values, in, out, 960, 9:16
python runner_detector.py

python runner_detector.py --input_dir in --output_dir out --min_height 960 --aspect_ratio 9:16
```

Command-Line Arguments
- --input_dir or -i: Directory containing input images (default: in).
- --output_dir or -o: Directory where output images will be saved (default: out).
- --min_height or -m: Minimum height (in pixels) for the cropped output (default: 960).
- --aspect_ratio or -r: Desired aspect ratio in WIDTH:HEIGHT format (default: 9:16).
- --watch or -w: If you run the script with the --watch (or -w) flag, the script uses Watchdog to monitor the in directory for new files. When a new image file is added, it automatically processes the image.

For example, to process images with a minimum height of 800 pixels and an aspect ratio of 16:9, use:

```bash
python runner_detector.py -i path/to/input -o path/to/output -m 800 -r 16:9
```

## Troubleshooting
GPU Issues:
If you have a compatible GPU, the script will use it for both detection and OCR. If not, it falls back to CPU. Ensure that your PyTorch installation matches your CUDA version if using a GPU.

SSL Certificate Issues:
The script includes a workaround to bypass SSL certificate issues when downloading the YOLO model.

## License
This project is licensed under the MIT License.