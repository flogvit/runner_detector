import cv2
import os
import ssl
import re
import torch
import easyocr
import argparse
import time
from ultralytics import YOLO

# Import watchdog modules
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold for OCR (numbers below this are ignored)


class RunnerDetector:
    def __init__(self, min_height=960, ratio_width=9, ratio_height=16):
        self.MIN_HEIGHT = min_height
        self.ratio_width = ratio_width
        self.ratio_height = ratio_height
        self.ASPECT_RATIO = ratio_width / ratio_height
        self.MIN_WIDTH = int(min_height * self.ASPECT_RATIO)

        print("Initializing models...")
        try:
            # Select device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Metal (MPS) for GPU acceleration")
            else:
                self.device = torch.device("cpu")
                print("No GPU available, using CPU")
                print(f"Number of CPU threads available: {os.cpu_count()}")

            # Load YOLO model
            self.model = YOLO('yolov8n.pt')
            if self.device.type == "mps":
                self.model.to(self.device)
            if self.device.type != "cpu":
                self.model.half()
                print("Converted YOLO model to half precision (FP16)")

            # Initialize EasyOCR
            use_gpu = torch.cuda.is_available()
            self.reader = easyocr.Reader(['en'], gpu=use_gpu, download_enabled=True, verbose=False)
            if use_gpu:
                print("Using CUDA GPU for OCR")
            else:
                print("Using CPU for OCR")

            # Decide a default batch size
            if self.device.type == "cuda":
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
                self.batch_size = min(max(int(gpu_mem / 2), 1), 8)
            elif self.device.type == "mps":
                self.batch_size = 16
            else:
                self.batch_size = 2

            print("Models initialized successfully!")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def detect_all_bib_numbers(self, image, person_bbox, horizontal_padding=0.3):
        """
        Detects *all* possible bib numbers (3-5 digits) in the ROI.
        Returns a list of (bib_string, confidence) sorted by confidence descending.
        """
        try:
            x1, y1, x2, y2 = map(int, person_bbox)
            height, width = image.shape[:2]

            # Crop region (torso)
            torso_y1 = y1
            torso_y2 = int(y1 + (y2 - y1) * 0.95)
            torso_x1 = int(x1 - (x2 - x1) * horizontal_padding)
            torso_x2 = int(x2 + (x2 - x1) * horizontal_padding)

            # Clamp coordinates
            torso_y1 = max(0, torso_y1)
            torso_y2 = min(height, torso_y2)
            torso_x1 = max(0, torso_x1)
            torso_x2 = min(width, torso_x2)

            torso_roi = image[torso_y1:torso_y2, torso_x1:torso_x2]

            # Downscale large ROI
            max_dim = 800
            roi_h, roi_w = torso_roi.shape[:2]
            if max(roi_h, roi_w) > max_dim:
                scale = max_dim / max(roi_h, roi_w)
                torso_roi = cv2.resize(torso_roi, (int(roi_w * scale), int(roi_h * scale)))
                print(f"Downscaled ROI to {torso_roi.shape[1]}x{torso_roi.shape[0]} for faster OCR.")

            # OCR
            results = self.reader.readtext(
                torso_roi,
                allowlist='0123456789',
                paragraph=False,
                batch_size=1,
                width_ths=0.5,
                height_ths=0.5,
                mag_ratio=1.0,
                contrast_ths=0.2
            )

            # Gather all numbers (3-5 digits) above confidence threshold
            found_numbers = []
            for (_, text, prob) in results:
                num = re.sub(r'[^0-9]', '', text)
                if len(num) in [2, 3, 4, 5] and prob > CONFIDENCE_THRESHOLD:
                    found_numbers.append((num, prob))

            found_numbers.sort(key=lambda x: x[1], reverse=True)
            if found_numbers:
                print(f"Detected {len(found_numbers)} bib(s) in ROI (padding={horizontal_padding}): {found_numbers}")
            return found_numbers

        except Exception as e:
            print(f"Error in detect_all_bib_numbers: {e}")
            return []

    def get_crop(self, x1, y1, x2, y2, img_width, img_height):
        """
        Calculates a ratio crop area centered on a detection box, ensuring minimum size.
        """
        try:
            det_height = y2 - y1
            base_padding = det_height * 0.05
            crop_y1 = max(0, int(y1 - base_padding))
            crop_y2 = min(img_height, int(y2 + base_padding))

            crop_height = crop_y2 - crop_y1
            crop_width = int(crop_height * self.ASPECT_RATIO)

            center_x = (x1 + x2) / 2
            crop_x1 = max(0, int(center_x - crop_width / 2))
            crop_x2 = min(img_width, int(center_x + crop_width / 2))

            if crop_height < self.MIN_HEIGHT:
                scale = self.MIN_HEIGHT / crop_height
                new_height = self.MIN_HEIGHT
                new_width = int(crop_width * scale)
                center_x = (crop_x1 + crop_x2) / 2
                center_y = (crop_y1 + crop_y2) / 2
                crop_x1 = max(0, int(center_x - new_width / 2))
                crop_x2 = min(img_width, int(center_x + new_width / 2))
                crop_y1 = max(0, int(center_y - new_height / 2))
                crop_y2 = min(img_height, int(center_y + new_height / 2))

            return (crop_x1, crop_y1, crop_x2, crop_y2)
        except Exception as e:
            print(f"Error in get_crop: {e}")
            return None

    def process_directory(self, input_dir, output_dir, processed_files=None):
        """
        Processes all images in the input directory in batches (YOLO detection),
        then processes each bounding box sequentially and outputs the crop area only once.

        If a set 'processed_files' is provided, files already processed are skipped.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_paths = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if processed_files is not None:
            image_paths = [p for p in image_paths if os.path.basename(p) not in processed_files]

        if not image_paths:
            print("No new images found in the input directory.")
            return

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []

            for path in batch_paths:
                img = cv2.imread(path)
                if img is not None:
                    batch_images.append((path, img))
                else:
                    print(f"Could not read image: {path}")
                    if processed_files is not None:
                        processed_files.add(os.path.basename(path))

            if not batch_images:
                continue

            imgs = [img for _, img in batch_images]
            results = self.model(imgs, conf=CONFIDENCE_THRESHOLD)

            for (path, img), result in zip(batch_images, results):
                base_name = os.path.splitext(os.path.basename(path))[0]
                print(f"\nProcessing file: {os.path.basename(path)}")

                all_boxes = []
                for box in result.boxes:
                    if int(box.cls) == 0:  # Person class
                        xyxy = box.xyxy[0].cpu().numpy()
                        all_boxes.append(xyxy)

                all_boxes.sort(key=lambda x: x[0])
                print(f"Found {len(all_boxes)} people in the image")

                processed_crops = set()

                for idx, bbox in enumerate(all_boxes):
                    x1, y1, x2, y2 = map(int, bbox)
                    print(f"Processing person {idx + 1} at position ({x1}, {y1})")

                    all_bibs = self.detect_all_bib_numbers(img, (x1, y1, x2, y2), horizontal_padding=0.3)
                    if not all_bibs:
                        print(f"No bibs found for person {idx + 1}. Skipping.")
                        continue

                    unique_bibs = sorted({bib for bib, conf in all_bibs})
                    combined_bib = "-".join(unique_bibs)
                    print(f"Combined bib for person {idx + 1}: {combined_bib}")

                    crop_coords = self.get_crop(x1, y1, x2, y2, img.shape[1], img.shape[0])
                    if not crop_coords:
                        print("Failed to get crop coordinates.")
                        continue
                    if crop_coords in processed_crops:
                        print(f"Crop region {crop_coords} already processed. Skipping duplicate.")
                        continue
                    processed_crops.add(crop_coords)

                    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
                    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    output_filename = f"{base_name}_runner_{combined_bib}_{idx + 1}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, cropped)
                    print(f"Saved runner crop with bib(s) {combined_bib} to {output_path}")

                if processed_files is not None:
                    processed_files.add(os.path.basename(path))


# Watchdog event handler
class NewImageHandler(FileSystemEventHandler):
    def __init__(self, detector, input_dir, output_dir, processed_files):
        super().__init__()
        self.detector = detector
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processed_files = processed_files

    def is_file_stable(self, filepath, wait_time=1.0):  # Added self here
        initial_size = os.path.getsize(filepath)
        time.sleep(wait_time)
        return os.path.getsize(filepath) == initial_size

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = event.src_path
            print(f"New image detected: {filepath}")
            # Wait until file size is stable
            while not self.is_file_stable(filepath, wait_time=1):
                print(f"Waiting for {filepath} to be fully copied...")
                time.sleep(1)
            self.detector.process_directory(self.input_dir, self.output_dir, processed_files=self.processed_files)



def main():
    parser = argparse.ArgumentParser(description="Runner Detector with Multi-Bib OCR")
    parser.add_argument("--input_dir", "-i", default="in", help="Directory with input images.")
    parser.add_argument("--output_dir", "-o", default="out", help="Directory for output images.")
    parser.add_argument("--min_height", "-m", type=int, default=960,
                        help="Minimum height in pixels for cropped output.")
    parser.add_argument("--aspect_ratio", "-r", default="9:16", help="Aspect ratio in WIDTH:HEIGHT (default 9:16).")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Watch the input directory for new files continuously.")
    args = parser.parse_args()

    try:
        ratio_width, ratio_height = map(float, args.aspect_ratio.split(':'))
    except Exception as e:
        print(f"Error parsing aspect ratio: {e}")
        return

    print(f"Using min_height: {args.min_height}, aspect ratio: {ratio_width}:{ratio_height}")

    try:
        detector = RunnerDetector(min_height=args.min_height, ratio_width=ratio_width, ratio_height=ratio_height)
        if args.watch:
            print("Watch mode enabled. Monitoring directory for new files...")
            processed_files = set()
            event_handler = NewImageHandler(detector, args.input_dir, args.output_dir, processed_files)
            observer = Observer()
            observer.schedule(event_handler, args.input_dir, recursive=False)
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping watchdog...")
                observer.stop()
            observer.join()
        else:
            detector.process_directory(args.input_dir, args.output_dir)
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
