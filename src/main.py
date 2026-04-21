"""main.py — ALPR Desktop UI (Tkinter)
Upload an image or use the camera to detect plates.
Run from the project root:  python src/main.py
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Path setup — works whether you run from project root or from src/
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from detect import VehiclePlateDetector
from preprocess import preprocess_plate
from ocr import EdgeLPRNet
from utils import draw_overlays, draw_plate_result, format_ocr_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#21262d"
ACCENT    = "#238636"
ACCENT_HV = "#2ea043"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
RED_ACC   = "#f85149"
PLATE_CLR = "#3fb950"

PREVIEW_W = 700
PREVIEW_H = 480


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class ALPRApp(tk.Tk):
    def __init__(self, detector: VehiclePlateDetector, ocr_engine):
        super().__init__()
        self.detector   = detector
        self.ocr_engine = ocr_engine

        self.title("Parking ALPR — Local License Plate Recognition")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(900, 620)

        self._photo_ref = None
        self.cap = None
        self.is_streaming = False
        self.current_frame = None
        self._processing = False

        self._build_ui()
        self.update_idletasks()
        w, h = 1100, 680
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def destroy(self):
        self.is_streaming = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_PANEL, pady=14)
        hdr.pack(fill="x")

        tk.Label(
            hdr, text="🚗  Parking ALPR System",
            font=("Segoe UI", 18, "bold"),
            fg=TEXT_PRI, bg=BG_PANEL,
        ).pack()
        tk.Label(
            hdr, text="YOLOv8 · LPRNet TFLite · Philippine Plates",
            font=("Segoe UI", 9),
            fg=TEXT_SEC, bg=BG_PANEL,
        ).pack()

        # ── Body ────────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DARK, padx=18, pady=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1, minsize=280)
        body.rowconfigure(0, weight=1)

        # ── Left: image pane ─────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG_PANEL, bd=1, relief="solid")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        tk.Label(
            left, text="Captured Frame",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_SEC, bg=BG_PANEL, pady=6,
        ).pack()

        self.img_label = tk.Label(
            left, bg="#010409",
            text="Upload an image or start the camera.",
            fg="#30363d", font=("Segoe UI", 12),
            compound="center",
        )
        self.img_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ── Right: controls + results ────────────────────────────────────────
        right = tk.Frame(body, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")

        # Upload Image button (primary)
        self.btn_upload = tk.Button(
            right,
            text="📁   Upload Image",
            font=("Segoe UI", 13, "bold"),
            fg="white", bg=ACCENT,
            activebackground=ACCENT_HV, activeforeground="white",
            relief="flat", bd=0,
            padx=16, pady=14,
            cursor="hand2",
            command=self._on_upload,
        )
        self.btn_upload.pack(fill="x", pady=(0, 8))
        self._bind_hover(self.btn_upload, ACCENT_HV, ACCENT)

        # Separator label
        tk.Label(
            right, text="— or use camera —",
            font=("Segoe UI", 9), fg="#484f58", bg=BG_DARK,
        ).pack(pady=(0, 8))

        # Start / Stop Camera button
        self.btn_camera = tk.Button(
            right,
            text="📷   Start Camera",
            font=("Segoe UI", 11, "bold"),
            fg="white", bg="#30363d",
            activebackground="#484f58", activeforeground="white",
            relief="flat", bd=0,
            padx=14, pady=10,
            cursor="hand2",
            command=self._on_toggle_camera,
        )
        self.btn_camera.pack(fill="x", pady=(0, 6))
        self._bind_hover(self.btn_camera, "#484f58", "#30363d")

        # Capture button (only active when camera is streaming)
        self.btn_capture = tk.Button(
            right,
            text="⏺   Capture & Detect",
            font=("Segoe UI", 11, "bold"),
            fg="white", bg="#21262d",
            activebackground="#484f58", activeforeground="white",
            relief="flat", bd=0,
            padx=14, pady=10,
            cursor="hand2",
            state="disabled",
            command=self._on_capture,
        )
        self.btn_capture.pack(fill="x")
        self._bind_hover(self.btn_capture, "#484f58", "#21262d")

        # Status label
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            right, textvariable=self.status_var,
            font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_DARK,
            wraplength=240, justify="left",
        ).pack(anchor="w", pady=(6, 0))

        tk.Frame(right, bg="#30363d", height=1).pack(fill="x", pady=14)

        # Plate results section
        tk.Label(
            right, text="Detected Plates",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_PRI, bg=BG_DARK, anchor="w",
        ).pack(fill="x")

        self.plates_frame = tk.Frame(right, bg=BG_DARK)
        self.plates_frame.pack(fill="x", pady=(6, 0))

        self._empty_label = tk.Label(
            self.plates_frame, text="—",
            font=("Segoe UI", 10), fg=TEXT_SEC, bg=BG_DARK,
        )
        self._empty_label.pack(anchor="w")

        tk.Frame(right, bg="#30363d", height=1).pack(fill="x", pady=14)

        # All detections log
        tk.Label(
            right, text="All Detections",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_PRI, bg=BG_DARK, anchor="w",
        ).pack(fill="x")

        self.det_text = tk.Text(
            right, height=9,
            bg=BG_CARD, fg="#79c0ff",
            font=("Consolas", 9),
            bd=0, relief="flat",
            insertbackground=TEXT_PRI,
            state="disabled",
        )
        self.det_text.pack(fill="both", expand=True, pady=(6, 0))

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _bind_hover(widget, hover_color, normal_color):
        widget.bind("<Enter>", lambda _: widget.config(bg=hover_color))
        widget.bind("<Leave>", lambda _: widget.config(bg=normal_color))

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    # ── Upload image ──────────────────────────────────────────────────────────
    def _on_upload(self):
        if self._processing:
            return
        path = filedialog.askopenfilename(
            title="Select an image to analyse",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")],
        )
        if not path:
            return

        # Stop camera if running
        self._stop_camera()

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Read Error", f"Cannot read image:\n{path}")
            return

        # Show the raw uploaded image immediately
        self._display_frame(frame)
        self._set_status(f"Processing {os.path.basename(path)}…")
        self._set_processing(True)
        threading.Thread(target=self._pipeline, args=(frame,), daemon=True).start()

    # ── Camera streaming ─────────────────────────────────────────────────────
    def _on_toggle_camera(self):
        if self.is_streaming:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam (device 0).")
            return
        self.is_streaming = True
        self._set_status("Live camera feed active.")
        self.btn_camera.config(text="⏹   Stop Camera")
        self.btn_capture.config(state="normal", bg="#30363d")
        self._update_stream()

    def _stop_camera(self):
        self.is_streaming = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.current_frame = None
        self.btn_camera.config(text="📷   Start Camera")
        self.btn_capture.config(state="disabled", bg="#21262d")

    def _update_stream(self):
        if self.is_streaming and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self._display_frame(frame)
            self.after(30, self._update_stream)

    def _display_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        resampling = getattr(Image, "Resampling", Image)
        lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
        img.thumbnail((PREVIEW_W, PREVIEW_H), lanczos)
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo, text="")
        self._photo_ref = photo

    # ── Capture pipeline ─────────────────────────────────────────────────────
    def _on_capture(self):
        if not self.is_streaming or self.current_frame is None or self._processing:
            return
        self._stop_camera()
        frame_to_process = self.current_frame.copy()
        self._display_frame(frame_to_process)
        self._set_status("Processing captured frame…")
        self._set_processing(True)
        threading.Thread(target=self._pipeline, args=(frame_to_process,), daemon=True).start()

    def _set_processing(self, active: bool):
        self._processing = active
        if active:
            self.btn_upload.config(state="disabled", text="⏳   Processing…")
            self.btn_capture.config(state="disabled")
        else:
            self.btn_upload.config(state="normal", text="📁   Upload Image")
            # Re-enable capture only if camera is streaming
            if self.is_streaming:
                self.btn_capture.config(state="normal", bg="#30363d")

    def _pipeline(self, frame):
        try:
            self.after(0, lambda: self._set_status("Running YOLO detection…"))

            # 2. YOLO ─────────────────────────────────────────────────────────
            detections, valid_plates = self.detector.process_frame(frame)
            annotated = draw_overlays(frame.copy(), detections)

            # 3. OCR on each plate crop ───────────────────────────────────────
            plate_results = []
            if self.ocr_engine and valid_plates:
                self.after(0, lambda: self._set_status(
                    f"Running OCR on {len(valid_plates)} plate(s)…"))

                for plate in valid_plates:
                    x1, y1, x2, y2 = plate["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    tensor   = preprocess_plate(crop)
                    ocr_out  = self.ocr_engine.extract_text(tensor)
                    # extract_text now returns a dict
                    raw_text = ocr_out["text"] if isinstance(ocr_out, dict) else ocr_out
                    conf     = ocr_out.get("confidence", 0.0) if isinstance(ocr_out, dict) else 0.0
                    result   = format_ocr_result(raw_text)
                    result["ocr_confidence"] = conf
                    plate_results.append(result)
                    annotated = draw_plate_result(annotated, plate["bbox"],
                                                  result["plate_number"])

            # 4. Update UI on the main thread ─────────────────────────────────
            self.after(0, lambda: self._update_ui(annotated, detections, plate_results))

        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Pipeline Error", str(exc)))
        finally:
            self.after(0, lambda: self._set_processing(False))

    # ── UI update (always on main thread) ────────────────────────────────────
    def _update_ui(self, annotated_bgr, detections, plate_results):
        # ── Frame image ──────────────────────────────────────────────────────
        rgb  = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        
        resampling = getattr(Image, "Resampling", Image)
        lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
        
        img.thumbnail((PREVIEW_W, PREVIEW_H), lanczos)
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo, text="")
        self._photo_ref = photo   # prevent GC

        # ── Plate results cards ──────────────────────────────────────────────
        for w in self.plates_frame.winfo_children():
            w.destroy()

        if plate_results:
            for r in plate_results:
                card = tk.Frame(self.plates_frame, bg=BG_CARD, pady=10, padx=14)
                card.pack(fill="x", pady=4)

                conf = r.get("ocr_confidence", 0.0)
                # Color-code by confidence: green > 70%, yellow > 40%, red otherwise
                if conf >= 0.7:
                    conf_color = PLATE_CLR  # green
                elif conf >= 0.4:
                    conf_color = "#f0c040"  # yellow
                else:
                    conf_color = RED_ACC    # red

                tk.Label(
                    card,
                    text=r["plate_number"],
                    font=("Consolas", 22, "bold"),
                    fg=conf_color, bg=BG_CARD,
                ).pack(anchor="w")

                conf_pct = f"{conf * 100:.1f}%"
                valid_tag = " \u2713 Valid PH" if r.get("valid") else ""
                tk.Label(
                    card,
                    text=f"Confidence: {conf_pct}{valid_tag}",
                    font=("Segoe UI", 9),
                    fg=TEXT_SEC, bg=BG_CARD,
                ).pack(anchor="w")
        else:
            tk.Label(
                self.plates_frame,
                text="No plates detected in this frame.",
                font=("Segoe UI", 10), fg=TEXT_SEC, bg=BG_DARK,
            ).pack(anchor="w")

        # ── Detections log ───────────────────────────────────────────────────
        self.det_text.config(state="normal")
        self.det_text.delete("1.0", "end")
        if detections:
            for d in detections:
                self.det_text.insert(
                    "end",
                    f"  {d['class_name']:<16}  conf={d['conf']:.3f}"
                    f"  bbox={d['bbox']}\n"
                )
        else:
            self.det_text.insert("end", "  No objects detected.\n")
        self.det_text.config(state="disabled")

        # ── Status ───────────────────────────────────────────────────────────
        n_obj = len(detections)
        n_plt = len(plate_results)
        self._set_status(
            f"Done — {n_obj} object(s) detected, {n_plt} plate(s) recognised."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    yolo_path = os.path.join(_PROJECT_ROOT, "models", "yolo26_custom.pt")
    lpr_path  = os.path.join(_PROJECT_ROOT, "models", "lprnet_ph_yolo_preprocessed.tflite")

    print("Loading models…")
    try:
        detector = VehiclePlateDetector(model_path=yolo_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    ocr_engine = None
    try:
        ocr_engine = EdgeLPRNet(model_path=lpr_path)
        print("[INFO] LPRNet loaded.")
    except Exception as e:
        print(f"[WARN] LPRNet not loaded ({e}). OCR will be skipped.")

    app = ALPRApp(detector, ocr_engine)
    app.mainloop()


if __name__ == "__main__":
    main()