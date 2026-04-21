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
        
        # Exact vocabulary from LPRnet_separable.py (I included, only O excluded)
        self.CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
        self.BLANK_IDX = len(self.CHARS)  # Index 35 is the CTC Blank

    def _decode_ctc(self, preds):
        """
        Greedy CTC Decoder with per-character confidence scoring.

        Parameters
        ----------
        preds : np.ndarray, shape [1, sequence_length, NUM_CLASS]
            Softmax probabilities from the model.

        Returns
        -------
        text : str
            Decoded plate string.
        char_confs : list[float]
            Per-character confidence (max softmax probability).
        avg_conf : float
            Average confidence across all decoded characters.
        """
        probs = preds[0]  # shape: [sequence_length, NUM_CLASS]
        char_indices = np.argmax(probs, axis=1)
        char_probs = np.max(probs, axis=1)

        text = ""
        char_confs = []
        prev_idx = -1

        for i, idx in enumerate(char_indices):
            # Skip CTC blank token and consecutive duplicates
            if idx != self.BLANK_IDX and idx != prev_idx:
                if idx < len(self.CHARS):
                    text += self.CHARS[idx]
                    char_confs.append(float(char_probs[i]))
            prev_idx = idx

        avg_conf = float(np.mean(char_confs)) if char_confs else 0.0
        return text, char_confs, avg_conf

    def extract_text(self, preprocessed_tensor):
        """
        Runs local TFLite inference on the preprocessed tensor.

        Returns
        -------
        dict with keys:
            text : str — decoded plate string
            confidence : float — average per-character confidence (0.0–1.0)
            char_confidences : list[float] — per-character confidence values
            raw_preds : np.ndarray — raw softmax output for diagnostics
        """
        try:
            # Set the tensor to point to the input data to be inferred
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_tensor)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get the output Softmax probabilities
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Decode with confidence scoring
            result_text, char_confs, avg_conf = self._decode_ctc(preds)
            
            # Debug logging
            print(f"[OCR] Decoded: '{result_text}' | "
                  f"Avg Conf: {avg_conf:.3f} | "
                  f"Chars: {len(result_text)} | "
                  f"Per-char: {[f'{c:.2f}' for c in char_confs]}")

            if not result_text:
                return {
                    "text": "UNKNOWN",
                    "confidence": 0.0,
                    "char_confidences": [],
                    "raw_preds": preds,
                }

            return {
                "text": result_text,
                "confidence": avg_conf,
                "char_confidences": char_confs,
                "raw_preds": preds,
            }

        except Exception as e:
            print(f"[Local OCR Error] Failed to process plate: {e}")
            return {
                "text": "ERROR",
                "confidence": 0.0,
                "char_confidences": [],
                "raw_preds": None,
            }