"""
LPRNet Diagnostic Script
=========================
Generates a synthetic plate with known ground truth, runs TFLite inference,
and visualizes the raw softmax output to confirm whether the model is
undertrained (flat distributions) or has a decoder bug (peaky but wrong).

Usage:
    python diagnose_lprnet.py
"""
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ── Resolve paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "lprnet_ph_yolo_preprocessed.tflite")
CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"  # 35 chars (O excluded)
BLANK_IDX = len(CHARS)  # index 35 = CTC blank
NUM_CLASSES = len(CHARS) + 1  # 36

# ── Preprocessing (must match training exactly) ───────────────────────────────
def preprocess_plate(image, target_size=(94, 24)):
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    sharpened = cv2.addWeighted(enhanced_gray, 1.5, blurred, -0.5, 0)
    
    processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    processed = processed.astype(np.float32) / 255.0
    tensor_input = np.expand_dims(processed, axis=0)
    return tensor_input, sharpened  # Return both tensor and visual


def create_synthetic_plate(text="ABC1234"):
    """Creates a simple synthetic plate image with known text for testing."""
    width, height = 300, 80
    plate = np.ones((height, width, 3), dtype=np.uint8) * 30  # Dark background
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 3
    
    # Center the text
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_w) // 2
    y = (height + text_h) // 2
    
    cv2.putText(plate, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return plate


def greedy_ctc_decode(preds):
    """Greedy CTC decode with confidence scores per character."""
    char_indices = np.argmax(preds[0], axis=1)
    char_probs = np.max(preds[0], axis=1)
    
    decoded_chars = []
    decoded_confs = []
    prev_idx = -1
    
    for i, idx in enumerate(char_indices):
        if idx != BLANK_IDX and idx != prev_idx:
            if idx < len(CHARS):
                decoded_chars.append(CHARS[idx])
                decoded_confs.append(char_probs[i])
            else:
                decoded_chars.append(f"?{idx}")
                decoded_confs.append(char_probs[i])
        prev_idx = idx
    
    return ''.join(decoded_chars), decoded_confs


def run_diagnostic():
    """Main diagnostic routine."""
    print("=" * 70)
    print("  LPRNet Diagnostic Tool")
    print("=" * 70)
    
    # ── 1. Load model ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found: {MODEL_PATH}")
        return
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n[INFO] Model Details:")
    print(f"   Input  shape: {input_details[0]['shape']}  dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}  dtype: {output_details[0]['dtype']}")
    print(f"   CHARS length: {len(CHARS)}  NUM_CLASSES: {NUM_CLASSES}")
    print(f"   BLANK_IDX: {BLANK_IDX}")
    
    expected_out = output_details[0]['shape']
    n_timesteps = expected_out[1]
    n_classes = expected_out[2]
    print(f"   Timesteps: {n_timesteps}  Classes: {n_classes}")
    
    if n_classes != NUM_CLASSES:
        print(f"\n[WARNING] CLASS COUNT MISMATCH! Model has {n_classes} classes, code expects {NUM_CLASSES}")
    
    # ── 2. Test with synthetic plates ──────────────────────────────────────
    test_cases = ["ABC1234", "XYZ789", "NCA5432", "TUV0123"]
    
    fig, axes = plt.subplots(len(test_cases), 3, figsize=(18, 4 * len(test_cases)))
    fig.suptitle("LPRNet Diagnostic: Softmax Distributions", fontsize=16, fontweight='bold')
    
    for row, gt_text in enumerate(test_cases):
        print(f"\n{'-' * 60}")
        print(f"  Test Case {row+1}: Ground Truth = '{gt_text}'")
        print(f"{'-' * 60}")
        
        # Create and preprocess
        plate_img = create_synthetic_plate(gt_text)
        tensor, visual = preprocess_plate(plate_img)
        
        # Verify tensor
        print(f"   Tensor shape: {tensor.shape}  dtype: {tensor.dtype}")
        print(f"   Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"   Tensor mean:  {tensor.mean():.4f}  std: {tensor.std():.4f}")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   Output shape: {preds.shape}")
        print(f"   Output range: [{preds.min():.6f}, {preds.max():.6f}]")
        
        # Decode
        decoded_text, confidences = greedy_ctc_decode(preds)
        avg_conf = np.mean(confidences) if confidences else 0
        
        print(f"   Decoded: '{decoded_text}' (avg confidence: {avg_conf:.4f})")
        
        # Per-timestep analysis
        argmax_indices = np.argmax(preds[0], axis=1)
        max_probs = np.max(preds[0], axis=1)
        blank_probs = preds[0][:, BLANK_IDX] if BLANK_IDX < n_classes else np.zeros(n_timesteps)
        
        print(f"\n   Per-timestep breakdown (top class, probability, blank prob):")
        # Only print first 5 and last 3 timesteps to reduce output noise
        print_indices = list(range(min(5, n_timesteps))) + list(range(max(n_timesteps-3, 5), n_timesteps))
        for t in range(n_timesteps):
            if t not in print_indices:
                if t == min(5, n_timesteps):
                    print(f"     ... ({n_timesteps - 8} timesteps omitted) ...")
                continue
            idx = argmax_indices[t]
            prob = max_probs[t]
            bp = blank_probs[t]
            char = CHARS[idx] if idx < len(CHARS) else "BLANK"
            marker = " <<" if idx != BLANK_IDX and prob > 0.5 else ""
            print(f"     t={t:2d}: '{char}' (idx={idx:2d}, p={prob:.4f}, blank={bp:.4f}){marker}")
        
        # Entropy analysis - how "confident" the model is
        entropy = -np.sum(preds[0] * np.log(preds[0] + 1e-10), axis=1)
        mean_entropy = np.mean(entropy)
        max_entropy = np.log(n_classes)  # Maximum possible entropy
        print(f"\n   Mean entropy: {mean_entropy:.4f} / {max_entropy:.4f} (max)")
        print(f"   Entropy ratio: {mean_entropy/max_entropy:.2%} (>80%% = nearly random = undertrained)")
        
        # ── Plot ───────────────────────────────────────────────────────────
        # Column 1: Original preprocessed plate
        axes[row, 0].imshow(visual, cmap='gray')
        axes[row, 0].set_title(f"GT: '{gt_text}'\nDecoded: '{decoded_text}'", fontsize=11)
        axes[row, 0].axis('off')
        
        # Column 2: Softmax heatmap
        im = axes[row, 1].imshow(preds[0].T, aspect='auto', cmap='hot', vmin=0, vmax=1)
        axes[row, 1].set_xlabel('Timestep')
        axes[row, 1].set_ylabel('Class')
        axes[row, 1].set_title(f'Softmax Heatmap (entropy={mean_entropy:.2f})', fontsize=11)
        # Add character labels on y-axis
        tick_labels = list(CHARS) + ['BLANK']
        if n_classes <= len(tick_labels):
            axes[row, 1].set_yticks(range(n_classes))
            axes[row, 1].set_yticklabels(tick_labels[:n_classes], fontsize=6)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.02)
        
        # Column 3: Max probability per timestep
        axes[row, 2].bar(range(n_timesteps), max_probs, color='steelblue', alpha=0.7, label='Max prob')
        axes[row, 2].bar(range(n_timesteps), blank_probs, color='red', alpha=0.4, label='Blank prob')
        axes[row, 2].axhline(y=1/n_classes, color='orange', linestyle='--', alpha=0.7, label=f'Random ({1/n_classes:.3f})')
        axes[row, 2].set_xlabel('Timestep')
        axes[row, 2].set_ylabel('Probability')
        axes[row, 2].set_title(f'Confidence per Timestep (avg={avg_conf:.3f})', fontsize=11)
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_path = os.path.join(SCRIPT_DIR, "diagnostic_output.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Diagnostic plot saved to: {output_path}")
    
    # Try to display
    try:
        plt.show()
    except Exception:
        print(f"   (Could not display plot window -- check the saved PNG)")
    
    # ── 3. Test with real images if available ──────────────────────────────
    real_dirs = [
        os.path.join(SCRIPT_DIR, "valid_preprocessed"),
        os.path.join(SCRIPT_DIR, "valid_cropped"),
        os.path.join(SCRIPT_DIR, "test_images"),
    ]
    
    for d in real_dirs:
        if os.path.isdir(d):
            images = [f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                print(f"\n{'=' * 60}")
                print(f"  Real images from: {d}")
                print(f"  Found {len(images)} images, testing first 10...")
                print(f"{'=' * 60}")
                
                for img_name in images[:10]:
                    img_path = os.path.join(d, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    tensor, _ = preprocess_plate(img)
                    interpreter.set_tensor(input_details[0]['index'], tensor)
                    interpreter.invoke()
                    preds = interpreter.get_tensor(output_details[0]['index'])
                    
                    decoded, confs = greedy_ctc_decode(preds)
                    avg_c = np.mean(confs) if confs else 0
                    
                    entropy = -np.sum(preds[0] * np.log(preds[0] + 1e-10), axis=1)
                    ent_ratio = np.mean(entropy) / np.log(n_classes)
                    
                    status = "[OK]" if avg_c > 0.5 and ent_ratio < 0.5 else "[!!]"
                    print(f"   {status} {img_name:30s} -> '{decoded:10s}' conf={avg_c:.3f} entropy={ent_ratio:.1%}")
    
    print(f"\n{'=' * 70}")
    print("  Diagnostic complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()
