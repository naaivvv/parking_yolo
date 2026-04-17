import cv2
import numpy as np

def preprocess_plate(image, target_size=(94, 24)):
    """
    Preprocesses the cropped plate specifically for the Keras LPRNet.
    target_size is (width, height) for OpenCV.
    """
    # 1. Resize strictly to 94 (Width) x 24 (Height)
    processed = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert BGR (OpenCV default) to RGB
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize pixel values to [0, 1] as expected by standard Keras floats
    processed = processed.astype(np.float32) / 255.0
    
    # 4. Add Batch dimension. 
    # Resulting shape: [1, 24, 94, 3] (Batch, Height, Width, Channels)
    tensor_input = np.expand_dims(processed, axis=0)
    
    return tensor_input