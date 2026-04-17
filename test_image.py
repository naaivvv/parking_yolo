"""
test_image.py — Static image ALPR test script
==============================================
Usage (from the project root):

    python test_image.py path/to/image.jpg
    python test_image.py path/to/image.jpg --lpr models/ph001.tflite
    python test_image.py path/to/image.jpg --no-ocr   # YOLO only, skip OCR

Output:
  - Prints all detections and plate OCR results to the console.
  - Shows an annotated window (press any key to close).
  - Saves the annotated image as test_output.jpg in the project root.
"""

import argparse
import os
import sys
import cv2

# ---------------------------------------------------------------------------
# Resolve project root and add src/ to path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Support running from project root OR from src/
if os.path.basename(_SCRIPT_DIR) == "src":
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
else:
    _PROJECT_ROOT = _SCRIPT_DIR

_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from detect import VehiclePlateDetector
from preprocess import preprocess_plate
from ocr import EdgeLPRNet
from utils import draw_overlays, draw_plate_result, format_ocr_result

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the YOLO + LPRNet pipeline on a static image."
    )
    parser.add_argument(
        "image",
        help="Path to the input image (JPG, PNG, BMP, etc.)"
    )
    parser.add_argument(
        "--yolo",
        default=os.path.join(_PROJECT_ROOT, "models", "yolo26_custom.pt"),
        help="Path to the YOLO .pt model file (default: models/yolo26_custom.pt)"
    )
    parser.add_argument(
        "--lpr",
        default=os.path.join(_PROJECT_ROOT, "models", "recognition.tflite"),
        help="Path to the TFLite LPRNet model (default: models/recognition.tflite)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR step — show YOLO detections only"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_PROJECT_ROOT, "test_output.jpg"),
        help="Where to save the annotated output image (default: test_output.jpg)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a display window (useful in headless environments)"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_test(args):
    # ── Load image ──────────────────────────────────────────────────────────
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[ERROR] OpenCV could not read image: {args.image}")
        sys.exit(1)

    print(f"[INFO] Image loaded: {args.image}  ({frame.shape[1]}x{frame.shape[0]})")

    # ── Load YOLO ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {args.yolo}")
    try:
        detector = VehiclePlateDetector(model_path=args.yolo, conf=args.conf)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Load LPRNet (optional) ───────────────────────────────────────────────
    ocr_engine = None
    if not args.no_ocr:
        print(f"[INFO] Loading LPRNet TFLite model: {args.lpr}")
        try:
            ocr_engine = EdgeLPRNet(model_path=args.lpr)
            print("[INFO] LPRNet loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load LPRNet ({e}). Running YOLO-only.")
            ocr_engine = None

    # ── Run YOLO ─────────────────────────────────────────────────────────────
    print("\n[INFO] Running YOLO inference...")
    detections, valid_plates = detector.process_frame(frame)

    print(f"\n{'─'*50}")
    print(f"  Total detections : {len(detections)}")
    print(f"  Plates found     : {len(valid_plates)}")
    print(f"{'─'*50}")
    for i, det in enumerate(detections):
        print(
            f"  [{i+1}] {det['class_name']:15s} "
            f"conf={det['conf']:.3f}  bbox={det['bbox']}"
        )
    print(f"{'─'*50}\n")

    # ── Run OCR on each plate crop ────────────────────────────────────────────
    annotated = draw_overlays(frame.copy(), detections)

    if ocr_engine and valid_plates:
        print("[INFO] Running OCR on detected plates...\n")
        for i, plate in enumerate(valid_plates):
            x1, y1, x2, y2 = plate["bbox"]
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                print(f"  Plate {i+1}: empty crop — skipped")
                continue

            tensor_input = preprocess_plate(plate_crop)
            
            # Extract and save the preprocessed crop being fed to the LPRNet model
            prep_img = (tensor_input[0] * 255.0).astype("uint8")
            prep_img_bgr = cv2.cvtColor(prep_img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(_PROJECT_ROOT, f"preprocessed_plate_{i+1}.jpg")
            cv2.imwrite(save_path, prep_img_bgr)
            print(f"  [DEBUG] Saved preprocessed crop to {save_path}")

            raw_text     = ocr_engine.extract_text(tensor_input)
            result       = format_ocr_result(raw_text)

            print(
                f"  Plate {i+1}: {result['plate_number']:12s} "
                f"| Valid PH: {str(result['valid']):5s} "
                f"| Raw: '{result['raw']}'"
            )

            # Draw OCR result on the annotated frame
            annotated = draw_plate_result(annotated, plate["bbox"], result["plate_number"])

    elif valid_plates and not ocr_engine:
        print("[INFO] OCR skipped (model not loaded).")
    else:
        print("[INFO] No plate crops to run OCR on.")

    # ── Save output ───────────────────────────────────────────────────────────
    cv2.imwrite(args.output, annotated)
    print(f"\n[INFO] Annotated image saved to: {args.output}")

    # ── Display ───────────────────────────────────────────────────────────────
    if not args.no_show:
        # Scale down for display if image is very large
        h, w = annotated.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale  = max_dim / max(h, w)
            display = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        else:
            display = annotated

        cv2.imshow("ALPR Test Result (press any key to close)", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n[DONE]")


if __name__ == "__main__":
    run_test(parse_args())
