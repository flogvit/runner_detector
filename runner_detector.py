import cv2
import os
import ssl
import re
import torch
import easyocr
import argparse
import time
import numpy as np
import concurrent.futures
from ultralytics import YOLO
from datetime import datetime

# Import watchdog modules
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Fix SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

CONFIDENCE_THRESHOLD = 0.4  # Confidence threshold for OCR (numbers below this are ignored)

def timer(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        # Check if the first argument is self and if timing is enabled
        if args and hasattr(args[0], 'timing_enabled') and args[0].timing_enabled:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"⏱️ {func.__name__} took {elapsed:.2f} seconds")
            return result
        else:
            # Just call the function without timing
            return func(*args, **kwargs)
    return wrapper


class RunnerDetector:
    def __init__(self, min_height=960, ratio_width=9, ratio_height=16, timing_enabled=False):
        self.MIN_HEIGHT = min_height
        self.ratio_width = ratio_width
        self.ratio_height = ratio_height
        self.ASPECT_RATIO = ratio_width / ratio_height
        self.MIN_WIDTH = int(min_height * self.ASPECT_RATIO)
        self.timing_enabled = timing_enabled

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

            # Set up thread pool for parallel processing
            max_workers = os.cpu_count() or 4
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

            print("Models initialized successfully!")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    @timer
    def enhance_image_for_ocr(self, img):
        """
        Enhanced version with more preprocessing techniques for better OCR results.
        Optimized for speed with prioritized techniques and early returns.
        """
        try:
            # Create a copy to avoid modifying the original
            img_copy = img.copy()

            # Define max dimension for ROI preprocessing
            MAX_ENHANCE_DIM = 400  # Keep this small for faster processing

            # Downscale if image is too large
            h, w = img_copy.shape[:2]
            if max(h, w) > MAX_ENHANCE_DIM:
                scale = MAX_ENHANCE_DIM / max(h, w)
                img_copy = cv2.resize(img_copy, (int(w * scale), int(h * scale)))

            enhanced_versions = []

            # Convert to grayscale - this is used by most enhancements
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            
            # Prioritize the most effective enhancements (based on success rates)
            # Add grayscale first - often most effective
            enhanced_versions.append(gray)
            
            # Apply adaptive thresholding - very effective for bib numbers
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            enhanced_versions.append(thresh)
            
            # Apply Otsu's thresholding - good for high contrast bibs
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_versions.append(otsu)

            # Limit to 3 enhancements for smaller ROIs to improve speed
            if max(h, w) <= MAX_ENHANCE_DIM / 2:
                return enhanced_versions
                
            # Add more enhancements for larger ROIs since they likely contain more detail
            
            # Apply sharpening - helps with blurry images
            blur = cv2.GaussianBlur(gray, (0, 0), 3)
            sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
            enhanced_versions.append(sharp)

            # Only use these more intensive techniques for medium-sized images
            if max(h, w) <= MAX_ENHANCE_DIM:
                # Apply morphological operations to clean noise
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                enhanced_versions.append(morph)
            
                # Apply contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_contrast = clahe.apply(gray)
                enhanced_versions.append(enhanced_contrast)
            
            # Skip color channel separation and edge detection - these are less effective for bib detection
            # and significantly increase processing time

            return enhanced_versions

        except Exception as e:
            print(f"Error in enhance_image_for_ocr: {e}")
            return [img]  # Return original if enhancement fails

    def detect_bib_regions(self, img, person_bbox):
        """Focus on upper torso area where bibs are typically worn - optimized for speed"""
        x1, y1, x2, y2 = map(int, person_bbox)
        height, width = img.shape[:2]

        # Upper torso region (25-45% of body height from top)
        # This is the most likely area for bibs
        torso_y1 = int(y1 + (y2 - y1) * 0.25)
        torso_y2 = int(y1 + (y2 - y1) * 0.45)

        # More selective padding for narrower region
        padding = int((x2 - x1) * 0.15)  # Slightly increased for better coverage
        torso_x1 = max(0, x1 - padding)
        torso_x2 = min(width, x2 + padding)

        return [(torso_x1, torso_y1, torso_x2, torso_y2)]

    def validate_bib_number(self, bib_numbers):
        """More stringent bib validation that only keeps high-confidence detections"""
        # Increase confidence threshold significantly
        HIGH_CONFIDENCE = 0.85
        MEDIUM_CONFIDENCE = 0.7

        # Only process numbers with reasonable confidence
        filtered_bibs = []

        for bib, conf in bib_numbers:
            # Clean the bib string
            cleaned_bib = re.sub(r'[^0-9]', '', bib)

            # Skip if empty or too short
            if not cleaned_bib or len(cleaned_bib) < 3:
                continue

            # High confidence - keep it
            if conf > HIGH_CONFIDENCE:
                filtered_bibs.append((cleaned_bib, conf))
            # Medium confidence - only keep if it matches expected race number format
            elif conf > MEDIUM_CONFIDENCE and len(cleaned_bib) >= 3 and len(cleaned_bib) <= 5:
                filtered_bibs.append((cleaned_bib, conf))

        # If we have results, return only the highest confidence one
        if filtered_bibs:
            return [max(filtered_bibs, key=lambda x: x[1])]

        return []

    @timer
    def detect_all_bib_numbers(self, image, person_bbox, horizontal_padding=0.3):
        """
        Detects all possible bib numbers in the ROI using multiple strategies.
        Returns a list of (bib_string, confidence) sorted by confidence.
        """
        try:
            all_results = []
            # Only use one padding value to save processing time
            # Define max dimension for scaling - smaller value will process faster
            MAX_DIM = 500  # Further reduced for faster processing

            # Special enhancement for yellow bibs (common in race bibs)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([40, 255, 255])

            # Use position-based detection instead of color-based
            bib_regions = self.detect_bib_regions(image, person_bbox)

            # Also try yellow color detection (specific for race bibs)
            x1, y1, x2, y2 = map(int, person_bbox)
            height, width = image.shape[:2]

            # Focus on torso region
            torso_y1 = y1
            torso_y2 = int(y1 + (y2 - y1) * 0.6)  # Upper torso only
            torso_x1 = max(0, x1 - int((x2 - x1) * 0.2))
            torso_x2 = min(width, x2 + int((x2 - x1) * 0.2))

            # Ensure valid coordinates
            torso_y1 = max(0, torso_y1)
            torso_y2 = min(height, torso_y2)

            if torso_y2 > torso_y1 and torso_x2 > torso_x1:
                torso_roi = image[torso_y1:torso_y2, torso_x1:torso_x2]

                # Convert to HSV for color detection
                hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
                yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

                # Find contours in the yellow mask
                contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Add yellow regions to bib_regions
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000 and area < 20000:  # Reasonable bib sizes
                        x, y, w, h = cv2.boundingRect(contour)
                        if 0.5 < w / h < 2.5:  # Reasonable aspect ratio for bib
                            # Convert back to full image coordinates
                            bib_regions.append((
                                x + torso_x1,
                                y + torso_y1,
                                x + w + torso_x1,
                                y + h + torso_y1
                            ))

            # Process detected bib regions if any
            if bib_regions:
                for region in bib_regions:
                    bx1, by1, bx2, by2 = region
                    # Ensure coordinates are within image
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(width, bx2), min(height, by2)

                    if bx2 <= bx1 or by2 <= by1:
                        continue

                    bib_roi = image[by1:by2, bx1:bx2]

                    # Skip empty ROIs
                    if bib_roi.size == 0:
                        continue

                    # OPTIMIZATION: Downscale ROI if larger than MAX_DIM
                    roi_h, roi_w = bib_roi.shape[:2]
                    if max(roi_h, roi_w) > MAX_DIM:
                        scale = MAX_DIM / max(roi_h, roi_w)
                        bib_roi = cv2.resize(bib_roi, (int(roi_w * scale), int(roi_h * scale)))

                    # Enhance image for better OCR
                    enhanced_versions = self.enhance_image_for_ocr(bib_roi)

                    # Try OCR on original and enhanced versions with early stopping
                    ocr_start = time.time() if self.timing_enabled else 0
                    HIGH_CONFIDENCE_THRESHOLD = 0.85  # Early stopping threshold
                    
                    # First check original image - often sufficient
                    results = self.reader.readtext(
                        bib_roi,
                        allowlist='0123456789',
                        paragraph=False,
                        batch_size=1,
                        width_ths=0.5,
                        height_ths=0.5
                    )
                    
                    found_high_confidence = False
                    for (_, text, prob) in results:
                        num = re.sub(r'[^0-9]', '', text)
                        if num and len(num) >= 3 and len(num) <= 5:  # Valid bib length
                            all_results.append((num, prob))
                            if prob > HIGH_CONFIDENCE_THRESHOLD:
                                found_high_confidence = True
                    
                    # Only try enhanced versions if needed
                    if not found_high_confidence:
                        for img_version in enhanced_versions:
                            results = self.reader.readtext(
                                img_version,
                                allowlist='0123456789',
                                paragraph=False,
                                batch_size=1,
                                width_ths=0.5,
                                height_ths=0.5
                            )
                            
                            for (_, text, prob) in results:
                                num = re.sub(r'[^0-9]', '', text)
                                if num and prob > CONFIDENCE_THRESHOLD:
                                    all_results.append((num, prob))
                                    if prob > HIGH_CONFIDENCE_THRESHOLD:
                                        found_high_confidence = True
                                        break
                            
                            if found_high_confidence:
                                break  # Stop processing more versions if high confidence result found
                                
                    if self.timing_enabled:
                        ocr_time = time.time() - ocr_start
                        print(f"⏱️ OCR for bib region took {ocr_time:.2f} seconds")

            # Fall back to scanning torso region - but only if needed and only with one padding
            if not all_results:
                # Use only one padding value to save time
                padding = 0.3  # Middle value is typically most effective
                
                # Define only the most promising region (upper body) for efficiency
                region = {
                    'y1': y1,
                    'y2': int(y1 + (y2 - y1) * 0.6),  # Upper body focus
                    'x1': int(x1 - (x2 - x1) * padding),
                    'x2': int(x2 + (x2 - x1) * padding)
                }
                
                # Clamp values to image bounds
                r_y1 = max(0, region['y1'])
                r_y2 = min(height, region['y2'])
                r_x1 = max(0, region['x1'])
                r_x2 = min(width, region['x2'])

                # Skip invalid regions
                if r_y2 > r_y1 and r_x2 > r_x1:
                    roi = image[r_y1:r_y2, r_x1:r_x2]

                    # Always downscale ROI for performance
                    roi_h, roi_w = roi.shape[:2]
                    if max(roi_h, roi_w) > MAX_DIM:
                        scale = MAX_DIM / max(roi_h, roi_w)
                        roi = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)))

                    # Get enhanced versions - but limit to essential ones
                    enhanced_versions = self.enhance_image_for_ocr(roi)

                    # Try OCR with early stopping when high confidence result found
                    ocr_start = time.time() if self.timing_enabled else 0
                    HIGH_CONFIDENCE_THRESHOLD = 0.85
                    found_high_confidence = False
                    
                    # Try original first
                    results = self.reader.readtext(
                        roi,
                        allowlist='0123456789',
                        paragraph=False,
                        batch_size=1,
                        width_ths=0.5,
                        height_ths=0.5,
                        mag_ratio=1.0,
                        contrast_ths=0.2
                    )

                    for (_, text, prob) in results:
                        num = re.sub(r'[^0-9]', '', text)
                        if num and len(num) >= 3 and len(num) <= 5:  # Valid bib length
                            all_results.append((num, prob))
                            if prob > HIGH_CONFIDENCE_THRESHOLD:
                                found_high_confidence = True
                    
                    # Only process enhanced versions if needed
                    if not found_high_confidence:
                        for img_version in enhanced_versions:
                            if found_high_confidence:
                                break
                                
                            results = self.reader.readtext(
                                img_version,
                                allowlist='0123456789',
                                paragraph=False,
                                batch_size=1,
                                width_ths=0.5,
                                height_ths=0.5,
                                mag_ratio=1.0,
                                contrast_ths=0.2
                            )

                            for (_, text, prob) in results:
                                num = re.sub(r'[^0-9]', '', text)
                                if num and prob > CONFIDENCE_THRESHOLD:
                                    all_results.append((num, prob))
                                    if prob > HIGH_CONFIDENCE_THRESHOLD:
                                        found_high_confidence = True
                                        break
                                        
                    if self.timing_enabled:
                        ocr_time = time.time() - ocr_start
                        print(f"⏱️ Fallback OCR for region took {ocr_time:.2f} seconds")

            # Validate and filter the detected numbers
            validated_numbers = self.validate_bib_number(all_results)

            if validated_numbers:
                print(f"Detected {len(validated_numbers)} valid bib(s): {validated_numbers}")

            return validated_numbers

        except Exception as e:
            print(f"Error in detect_all_bib_numbers: {e}")
            return []

    def get_crop(self, x1, y1, x2, y2, img_width, img_height):
        """
        Calculates a ratio crop area centered on a detection box, ensuring minimum size.
        Avoids partial crops by checking if the person is too close to the edge.
        """
        try:
            # Calculate detection box dimensions
            det_width = x2 - x1
            det_height = y2 - y1
            
            # Skip if the detection is at the edge of the image (likely a partial person)
            # Check if the detection box is very close to any edge
            edge_margin = 0.05  # 5% of image dimension
            img_edge_margin_x = img_width * edge_margin
            img_edge_margin_y = img_height * edge_margin
            
            # Skip if the person is cut off at the edge (detection box touches image edge)
            is_at_left_edge = x1 < img_edge_margin_x
            is_at_right_edge = x2 > (img_width - img_edge_margin_x)
            is_at_top_edge = y1 < img_edge_margin_y
            is_at_bottom_edge = y2 > (img_height - img_edge_margin_y)
            
            # If the detection is too close to edges, it might be a partial person
            # Only process detections that are not at image edges
            if is_at_left_edge or is_at_right_edge or is_at_top_edge or is_at_bottom_edge:
                person_width_ratio = det_width / img_width
                person_height_ratio = det_height / img_height
                
                # If the person takes up a significant portion of the image, we'll keep it
                # Otherwise, it's likely a partial person at the edge
                if (person_width_ratio < 0.4 and person_height_ratio < 0.6):
                    print(f"Skipping likely partial person at edge: ({x1}, {y1}, {x2}, {y2})")
                    return None
            
            # Normal cropping procedure
            base_padding = det_height * 0.05
            crop_y1 = max(0, int(y1 - base_padding))
            crop_y2 = min(img_height, int(y2 + base_padding))

            crop_height = crop_y2 - crop_y1
            crop_width = int(crop_height * self.ASPECT_RATIO)

            center_x = (x1 + x2) / 2
            crop_x1 = max(0, int(center_x - crop_width / 2))
            crop_x2 = min(img_width, int(center_x + crop_width / 2))

            # Ensure minimum size if needed
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

    @timer
    def process_runner_detection(self, img, bbox, base_name, idx, output_dir):
        """
        Process a single runner detection in parallel
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            print(f"Processing person {idx + 1} at position ({x1}, {y1})")

            # Get bibs with improved detection
            all_bibs = self.detect_all_bib_numbers(img, (x1, y1, x2, y2))
            if not all_bibs:
                print(f"No bibs found for person {idx + 1}. Skipping.")
                return None

            # Create crop (skipping partial people at image edges)
            crop_coords = self.get_crop(x1, y1, x2, y2, img.shape[1], img.shape[0])
            if not crop_coords:
                print(f"Skipping person {idx + 1} - likely partial or at edge")
                return None

            # Extract the crop
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
            
            # Additional check: if the crop is touching the image edges on 
            # multiple sides, it's likely a partial person
            edge_touches = 0
            if crop_x1 == 0:
                edge_touches += 1
            if crop_y1 == 0:
                edge_touches += 1
            if crop_x2 == img.shape[1]:
                edge_touches += 1
            if crop_y2 == img.shape[0]:
                edge_touches += 1
                
            # If touching more than one edge, likely a partial crop
            if edge_touches > 1:
                print(f"Skipping person {idx + 1} - crop touches {edge_touches} image edges")
                return None
                
            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # Prepare filename with all detected bibs
            unique_bibs = sorted({bib for bib, conf in all_bibs})
            combined_bib = "-".join(unique_bibs)
            print(f"Combined bib for person {idx + 1}: {combined_bib}")

            output_filename = f"{base_name}_runner_{combined_bib}_{idx + 1}.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # Save the image
            cv2.imwrite(output_path, cropped)
            print(f"Saved runner crop with bib(s) {combined_bib} to {output_path}")

            return (crop_coords, output_path)

        except Exception as e:
            print(f"Error in process_runner_detection: {e}")
            return None

    @timer
    def process_directory(self, input_dir, output_dir, processed_files=None):
        """
        Processes all images in the input directory in batches (YOLO detection),
        then processes each bounding box in parallel and outputs the crop area.

        If a set 'processed_files' is provided, files already processed are skipped.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        global_detected_bibs = set()  # Track bibs across all people
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

            # Pre-check image dimensions to adjust batch size if needed
            adjusted_batch_size = self.batch_size
            for path in batch_paths:
                try:
                    # Fast header-only read to get dimensions
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        h, w = img.shape[:2]
                        # If images are extremely large, reduce batch size
                        if w * h > 8000000:  # 8MP
                            adjusted_batch_size = max(1, self.batch_size // 2)
                            print(f"Large image detected ({w}x{h}), reducing batch size to {adjusted_batch_size}")
                            break
                except Exception as e:
                    print(f"Error checking image dimensions: {e}")

            # Use adjusted batch size
            batch_paths = batch_paths[:adjusted_batch_size]

            # Load images
            if self.timing_enabled:
                load_start = time.time()
            
            for path in batch_paths:
                img = cv2.imread(path)
                if img is not None:
                    batch_images.append((path, img))
                else:
                    print(f"Could not read image: {path}")
                    if processed_files is not None:
                        processed_files.add(os.path.basename(path))
                        
            if self.timing_enabled:
                load_time = time.time() - load_start
                print(f"⏱️ Image loading for {len(batch_paths)} files took {load_time:.2f} seconds")

            if not batch_images:
                continue

            imgs = [img for _, img in batch_images]

            # Run YOLO detection with adjusted confidence
            if self.timing_enabled:
                detection_start = time.time()
                results = self.model(imgs, conf=CONFIDENCE_THRESHOLD)
                detection_time = time.time() - detection_start
                print(f"⏱️ YOLO detection for {len(imgs)} images took {detection_time:.2f} seconds")
            else:
                results = self.model(imgs, conf=CONFIDENCE_THRESHOLD)

            for (path, img), result in zip(batch_images, results):
                base_name = os.path.splitext(os.path.basename(path))[0]
                print(f"\nProcessing file: {os.path.basename(path)}")

                # Extract boxes with confidence
                all_boxes = []
                for box in result.boxes:
                    if int(box.cls) == 0:  # Person class
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        # Only keep detections with good confidence
                        if conf >= CONFIDENCE_THRESHOLD:
                            # Add confidence as a 5th element
                            box_with_conf = np.append(xyxy, conf)
                            all_boxes.append(box_with_conf)

                # Filter out overlapping boxes
                filtered_boxes = self.filter_overlapping_boxes(all_boxes, iou_threshold=0.3)

                # Convert back to format without confidence
                filtered_boxes = [box[:4] for box in filtered_boxes]

                # Sort by x-coordinate
                filtered_boxes.sort(key=lambda x: x[0])
                print(f"Found {len(filtered_boxes)} people in the image after filtering")

                processed_crops = set()
                futures = []

                # Submit tasks to thread pool for parallel processing
                for idx, bbox in enumerate(filtered_boxes):
                    futures.append(
                        self.executor.submit(
                            self.process_runner_detection,
                            img, bbox, base_name, idx, output_dir
                        )
                    )

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            crop_coords, output_path = result  # Fix: Only 2 values returned

                            # Extract bib from filename
                            bib_info = os.path.basename(output_path)
                            bib_parts = bib_info.split('_')
                            if len(bib_parts) > 1:
                                bib_number = bib_parts[1]

                                # Skip if this bib was already detected on another person
                                if bib_number in global_detected_bibs and bib_number != "unknown":
                                    print(f"Skipping duplicate bib {bib_number}")
                                    continue

                                global_detected_bibs.add(bib_number)
                            processed_crops.add(crop_coords)
                    except Exception as e:
                        print(f"Error processing runner detection: {e}")
                        # Continue with next future instead of stopping

    def filter_overlapping_boxes(self, boxes, iou_threshold=0.3):
        """
        Filter out overlapping bounding boxes and edge detections
        that are likely partial people
        """
        if not boxes:
            return []

        # Sort boxes by confidence (highest first)
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        kept_boxes = []
        img_width, img_height = 0, 0
        
        # If we have at least one box, use its coordinates to estimate image dimensions
        if len(boxes) > 0:
            # Estimate image dimensions based on detections
            # This works because YOLO coordinates are in absolute pixels
            img_width = max([box[2] for box in boxes]) * 1.1  # Add margin
            img_height = max([box[3] for box in boxes]) * 1.1  # Add margin

        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2, conf = box
            
            # Skip boxes with very low confidence
            if conf < 0.4:  # More strict minimum confidence
                continue
                
            # Skip boxes that are very small
            width = x2 - x1
            height = y2 - y1
            if width < 50 or height < 100:  # Minimum reasonable size for a person
                continue
                
            # Skip boxes that are likely partial detections at image edges
            edge_margin = 0.03  # 3% of image dimension
            img_margin_x = img_width * edge_margin
            img_margin_y = img_height * edge_margin
            
            is_at_edge = (
                x1 < img_margin_x or 
                x2 > (img_width - img_margin_x) or
                y1 < img_margin_y or
                y2 > (img_height - img_margin_y)
            )
            
            # Only filter out edge detections that are small relative to the image
            # (likely partial people)
            if is_at_edge:
                person_width_ratio = width / img_width
                person_height_ratio = height / img_height
                
                # Skip if small and at edge (likely partial)
                if person_width_ratio < 0.25 and person_height_ratio < 0.4:
                    continue
            
            # Handle overlapping boxes
            should_keep = True
            for kept_box in kept_boxes:
                if self.calculate_iou(box[:4], kept_box[:4]) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                kept_boxes.append(box)

        return kept_boxes

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area

# Watchdog event handler
class NewImageHandler(FileSystemEventHandler):
    def __init__(self, detector, input_dir, output_dir, processed_files):
        super().__init__()
        self.detector = detector
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processed_files = processed_files

    def is_file_stable(self, filepath, wait_time=1.0):
        """Check if file has finished copying by comparing file sizes over time"""
        try:
            initial_size = os.path.getsize(filepath)
            time.sleep(wait_time)
            return os.path.getsize(filepath) == initial_size
        except Exception as e:
            print(f"Error checking file stability: {e}")
            return False

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = event.src_path
            print(f"New image detected: {filepath}")
            # Wait until file size is stable
            attempts = 0
            while not self.is_file_stable(filepath, wait_time=1) and attempts < 10:
                print(f"Waiting for {filepath} to be fully copied...")
                attempts += 1
                time.sleep(1)

            if attempts >= 10:
                print(f"Warning: File {filepath} may not be fully copied after 10 seconds.")

            self.detector.process_directory(self.input_dir, self.output_dir, processed_files=self.processed_files)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Runner Detector with Multi-Bib OCR")
    parser.add_argument("--input_dir", "-i", default="in", help="Directory with input images.")
    parser.add_argument("--output_dir", "-o", default="out", help="Directory for output images.")
    parser.add_argument("--min_height", "-m", type=int, default=960,
                        help="Minimum height in pixels for cropped output.")
    parser.add_argument("--aspect_ratio", "-r", default="9:16", help="Aspect ratio in WIDTH:HEIGHT (default 9:16).")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Watch the input directory for new files continuously.")
    parser.add_argument("--race_format", "-f", default=None,
                        help="Optional: Specific race format to optimize detection (marathon, 5k, etc.)")
    parser.add_argument("--timing", "-t", action="store_true",
                        help="Enable timing logs for performance analysis.")
    args = parser.parse_args()
    
    # Only start timing if enabled
    overall_start_time = time.time() if args.timing else None

    try:
        ratio_width, ratio_height = map(float, args.aspect_ratio.split(':'))
    except Exception as e:
        print(f"Error parsing aspect ratio: {e}")
        return

    print(f"Using min_height: {args.min_height}, aspect ratio: {ratio_width}:{ratio_height}")
    if args.race_format:
        print(f"Optimizing for race format: {args.race_format}")

    try:
        detector = RunnerDetector(min_height=args.min_height, ratio_width=ratio_width, ratio_height=ratio_height, timing_enabled=args.timing)
        if args.watch:
            print("Watch mode enabled. Monitoring directory for new files...")
            processed_files = set()
            event_handler = NewImageHandler(detector, args.input_dir, args.output_dir, processed_files)
            observer = Observer()
            observer.schedule(event_handler, args.input_dir, recursive=False)
            observer.start()
            try:
                # Process existing files first
                detector.process_directory(args.input_dir, args.output_dir, processed_files=processed_files)

                # Then watch for new files
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping watchdog...")
                observer.stop()
            finally:
                # Clean up thread pool
                detector.executor.shutdown()
                observer.join()
        else:
            detector.process_directory(args.input_dir, args.output_dir)
            # Clean up thread pool
            detector.executor.shutdown()
            
            # Print overall execution time if timing is enabled
            if args.timing:
                overall_time = time.time() - overall_start_time
                print(f"\n⏱️ Total execution time: {overall_time:.2f} seconds")
                print(f"⏱️ Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error in main: {e}")
        # Print execution time even if there was an error
        if args.timing and overall_start_time:
            overall_time = time.time() - overall_start_time
            print(f"\n⏱️ Execution time until error: {overall_time:.2f} seconds")


if __name__ == "__main__":
    main()