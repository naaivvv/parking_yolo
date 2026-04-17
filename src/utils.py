import cv2
import re

# Colour palette (BGR)
_COLORS = {
    "car":           (0, 200, 255),   # amber
    "large vehicle": (0, 100, 255),   # orange
    "motorcycle":    (255, 180, 0),   # light blue
    "plate":         (0, 255, 120),   # green
    "unknown":       (180, 180, 180), # grey
}

# Philippine plate pattern: 3 letters + 4 digits, plain form (e.g. ABC1234)
_PH_PLATE_RE = re.compile(r"^[A-Z]{3}[0-9]{4}$")


def draw_overlays(frame, detections: list) -> object:
    """
    Draws bounding boxes and class labels onto *frame* (BGR numpy array).

    Parameters
    ----------
    frame      : BGR numpy array (modified in-place)
    detections : list of dicts with keys: class_name, bbox, conf

    Returns
    -------
    Annotated frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det.get("class_name", "unknown")
        conf = det.get("conf", 0.0)
        color = _COLORS.get(cls, _COLORS["unknown"])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA
        )

    return frame


def draw_plate_result(frame, bbox: tuple, plate_text: str) -> object:
    """
    Overlays the recognised plate text beneath the plate bounding box.

    Parameters
    ----------
    frame      : BGR numpy array
    bbox       : (x1, y1, x2, y2)
    plate_text : decoded OCR string

    Returns
    -------
    Annotated frame.
    """
    x1, y1, x2, y2 = bbox
    label = f"[OCR] {plate_text}"
    color = (0, 255, 120)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 8), color, -1)
    cv2.putText(
        frame, label,
        (x1 + 3, y2 + th + 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 0, 0), 2, cv2.LINE_AA
    )
    return frame


def format_ocr_result(raw_text: str) -> dict:
    """
    Formats raw OCR output into a structured result dict.

    The returned plate_number contains ONLY uppercase letters and digits —
    no dashes, spaces, periods, or any other symbols.

    Returns
    -------
    dict with keys:
        plate_number : plain alphanumeric string  (e.g. "ABC1234")
        raw          : original decoded string from the model
        valid        : True if matches Philippine plate pattern (LLL-NNNN)
    """
    # Strip everything except uppercase letters and digits
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.strip().upper())

    # Validate against Philippine plate pattern (3 letters + 4 digits)
    valid = bool(_PH_PLATE_RE.match(cleaned))

    return {
        "plate_number": cleaned,   # plain, no separators
        "raw": raw_text,
        "valid": valid,
    }
