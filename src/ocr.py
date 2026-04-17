import numpy as np
import tensorflow as tf

class EdgeLPRNet:
    def __init__(self, model_path="models/ph001.tflite"):
        """
        Initializes the TensorFlow Lite Interpreter.
        TFLite is required for real-time performance on a Raspberry Pi.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Exact vocabulary from your LPRnet.py (Excludes I and O)
        self.CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
        self.BLANK_IDX = len(self.CHARS) # Index 35 is the CTC Blank

    def _decode_ctc(self, preds):
        """
        Greedy CTC Decoder tailored to Keras Softmax output.
        preds shape: [1, sequence_length, NUM_CLASS]
        """
        # Get the most likely character index per timestep
        char_indices = np.argmax(preds[0], axis=1)
        
        text = ""
        prev_idx = -1
        
        for idx in char_indices:
            # Skip CTC blank token and consecutive duplicates
            if idx != self.BLANK_IDX and idx != prev_idx:
                text += self.CHARS[idx]
            prev_idx = idx
            
        return text

    def extract_text(self, preprocessed_tensor):
        """Runs local TFLite inference on the preprocessed tensor."""
        try:
            # Set the tensor to point to the input data to be inferred
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_tensor)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output Softmax probabilities
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Decode to Philippine plate text
            result_text = self._decode_ctc(preds)
            
            if not result_text:
                return "UNKNOWN"
                
            return result_text
            
        except Exception as e:
            print(f"[Local OCR Error] Failed to process plate: {e}")
            return "ERROR"