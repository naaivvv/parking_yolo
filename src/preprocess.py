import cv2
import numpy as np

def preprocess_yolo(frame):
    """
    Preprocesses the full frame specifically for YOLO inference.
    (Ultralytics YOLO internally handles standard resizing and normalization).
    This function adds Gamma Correction to gently lift shadows and improve 
    vehicle/plate detection in under-exposed or harsh parking lot lighting,
    while preserving the full RGB color profile that YOLO relies on.
    """
    gamma = 1.25  # >1.0 brightens shadows, <1.0 darkens highlights
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply lookup table
    enhanced = cv2.LUT(frame, table)
    
    return enhanced


def preprocess_plate(image, target_size=(94, 24)):
    """
    Preprocesses the cropped plate specifically for the Keras LPRNet.
    target_size is (width, height) for OpenCV.
    Applies strong contrast enhancements (Grayscale + CLAHE + Sharpening).
    """
    # 1. Resize strictly to 94 (Width) x 24 (Height)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale to remove color noise and variability
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Using small tiles since the resolution (94x24) is quite small.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)
    
    # 4. Sharpen the image to make edges (text) more distinct
    blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    sharpened = cv2.addWeighted(enhanced_gray, 1.5, blurred, -0.5, 0)
    
    # Note: If you want strictly binary black and white, you can uncomment this:
    # _, sharpened = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Convert back to RGB for the 3-channel requirement of LPRNet
    processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    
    # 6. Normalize pixel values to [0, 1] as expected by standard Keras floats
    processed = processed.astype(np.float32) / 255.0
    
    # 7. Add Batch dimension. 
    # Resulting shape: [1, 24, 94, 3] (Batch, Height, Width, Channels)
    tensor_input = np.expand_dims(processed, axis=0)
    
    return tensor_input