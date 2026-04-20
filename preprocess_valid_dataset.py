import os
import cv2
import glob
import sys
import numpy as np

# Add src to python path to import from src.detect
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from detect import VehiclePlateDetector
from preprocess import preprocess_plate

def main():
    input_dir = os.path.join("Plate Recognition.v1i.coco", "valid")
    output_dir = "valid_preprocessed"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Initializing YOLO model...")
    detector = VehiclePlateDetector()
    
    # Supported image extensions
    valid_exts = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    processed_count = 0
    no_plate_count = 0
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Get detections
        detections, valid_plates = detector.process_frame(frame)
        
        if len(valid_plates) == 0:
            no_plate_count += 1
            print(f"No plate detected in {img_path}")
            continue
            
        # If multiple plates, take the one with highest confidence
        best_plate = max(valid_plates, key=lambda x: x["conf"])
        x1, y1, x2, y2 = best_plate["bbox"]
        
        # Ensure coordinates are within bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Avoid zero area crop
        if x2 <= x1 or y2 <= y1:
            no_plate_count += 1
            print(f"Invalid crop dimensions for {img_path}: {x1},{y1} to {x2},{y2}")
            continue

        cropped_plate = frame[y1:y2, x1:x2]
        
        # Apply LPRNet preprocessing to make it look like the synthetic dataset
        processed_tensor = preprocess_plate(cropped_plate) # shape: (1, 24, 94, 3), range: [0, 1]
        
        # Convert back to image format (0-255, uint8) for saving
        final_plate_img = (processed_tensor[0] * 255.0).astype(np.uint8)
        
        # Save the preprocessed plate
        base_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, base_name)
        
        cv2.imwrite(out_path, final_plate_img)
        processed_count += 1
        
    print(f"\nProcessing complete!")
    print(f"Successfully processed and cropped plates for {processed_count} images.")
    print(f"Failed to detect plates in {no_plate_count} images.")
    print(f"Cropped plates saved to: {output_dir}")

if __name__ == "__main__":
    main()
