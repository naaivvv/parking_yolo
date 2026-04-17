import os
import cv2
from ultralytics import YOLO

# Class indices as defined in the training dataset
CLASS_NAMES = ['car', 'large vehicle', 'motorcycle', 'plate']
PLATE_CLASS_IDX = 3       # 'plate' is index 3
VEHICLE_CLASS_IDXS = {0, 1, 2}  # car, large vehicle, motorcycle

# Resolve model path relative to the project root (one level above src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "yolo26_custom.pt")


class VehiclePlateDetector:
    """
    Wraps the custom-trained YOLOv8 model to detect vehicles and license plates
    in a single forward pass.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, conf: float = 0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found at: {model_path}\n"
                f"Place your model at '{model_path}' or pass the correct path."
            )
        print(f"[Detector] Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        print("[Detector] Model loaded successfully.")

    def process_frame(self, frame):
        """
        Runs YOLO inference on a single BGR frame.

        Returns
        -------
        detections : list[dict]
            All detected objects with keys: class_id, class_name, bbox, conf
        valid_plates : list[dict]
            Subset of detections where class == 'plate', each with key 'bbox'
        """
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        boxes = results[0].boxes

        detections = []
        valid_plates = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                det = {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
                    "bbox": (x1, y1, x2, y2),
                    "conf": confidence,
                }
                detections.append(det)

                if class_id == PLATE_CLASS_IDX:
                    valid_plates.append({"bbox": (x1, y1, x2, y2), "conf": confidence})

        return detections, valid_plates

    def annotate_frame(self, frame):
        """
        Convenience method — runs inference and returns the annotated frame
        directly from Ultralytics' built-in plot() method.
        """
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        return results[0].plot()
