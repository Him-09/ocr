import cv2
import numpy as np
from PIL import Image


def enhance_contrast(img_array):
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)

    # Local contrast enhancement
    kernel = np.ones((3, 3), np.float32) / 9
    local_contrast = cv2.filter2D(enhanced, -1, kernel)

    return enhanced - local_contrast


def detect_number_regions(img_array):
    # Add circular mask detection
    circles = cv2.HoughCircles(
        img_array,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )

    regions = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Extract region around each circle
            x, y, r = i[0], i[1], i[2]
            region = img_array[max(0, y - r):min(img_array.shape[0], y + r),
                     max(0, x - r):min(img_array.shape[1], x + r)]
            if region.size > 0:  # Only add non-empty regions
                regions.append(region)
    return regions


def preprocess_image(img):
    # Convert to grayscale if not already
    if img.mode != 'L':
        img = img.convert('L')

    # Convert to numpy array
    img_array = np.array(img)

    # Enhance contrast
    enhanced = enhance_contrast(img_array)

    # Detect potential number regions
    regions = detect_number_regions(enhanced)

    if regions:
        # Process each region
        processed_regions = []
        for region in regions:
            if region.size == 0:  # Skip empty regions
                continue

            # Apply adaptive thresholding
            try:
                binary = cv2.adaptiveThreshold(
                    region,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11,
                    2
                )

                # Remove noise
                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                # Find contours in this region
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Crop to the number
                    number_region = region[y:y + h, x:x + w]
                    if number_region.size > 0:  # Only add non-empty regions
                        processed_regions.append(number_region)
            except cv2.error:
                continue

        if processed_regions:
            # Use the region with the highest contrast
            best_region = max(processed_regions, key=lambda x: np.std(x))

            # Resize maintaining aspect ratio
            target_size = 64
            h, w = best_region.shape
            aspect_ratio = w / h
            if aspect_ratio > 1:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)

            resized = cv2.resize(best_region, (new_w, new_h))

            # Create a square image with padding
            square_img = np.ones((target_size, target_size), dtype=np.uint8) * 255
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            square_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            return Image.fromarray(square_img)

    # If no regions found, return a blank image
    blank_img = np.ones((64, 64), dtype=np.uint8) * 255
    return Image.fromarray(blank_img)