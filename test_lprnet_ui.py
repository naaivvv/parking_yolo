"""
test_lprnet_ui.py — LPRNet Plate OCR Tester (Tkinter UI)
=========================================================
Opens a dark-themed GUI that lets you:
  1. Select individual images (or a whole folder) of plate crops.
  2. Applies preprocess_plate() and runs the LPRNet TFLite model.
  3. Shows the original image, preprocessed tensor, and OCR result.

Usage:
    python test_lprnet_ui.py
    python test_lprnet_ui.py --model models/lprnet_ph_yolo_preprocessed.tflite
    python test_lprnet_ui.py --dir valid_preprocessed
"""

import argparse
import glob
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_SCRIPT_DIR) == "src":
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
else:
    _PROJECT_ROOT = _SCRIPT_DIR

_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from preprocess import preprocess_plate
from ocr import EdgeLPRNet
from utils import format_ocr_result

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_CARD    = "#21262d"
BG_INPUT   = "#0d1117"
ACCENT     = "#238636"
ACCENT_HV  = "#2ea043"
CYAN       = "#58a6ff"
TEXT_PRI   = "#e6edf3"
TEXT_SEC   = "#8b949e"
PLATE_CLR  = "#3fb950"
GOLD       = "#d29922"
RED_ACC    = "#f85149"
BORDER     = "#30363d"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_HEAD  = ("Segoe UI", 12, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_PLATE = ("Consolas", 28, "bold")
FONT_MONO  = ("Consolas", 10)
FONT_LOG   = ("Consolas", 9)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LPRNet Plate OCR Tester — visual GUI for testing plate recognition."
    )
    parser.add_argument(
        "--model",
        default=os.path.join(_PROJECT_ROOT, "models", "lprnet_ph_yolo_preprocessed.tflite"),
        help="Path to the TFLite LPRNet model.",
    )
    parser.add_argument(
        "--dir",
        default=os.path.join(_PROJECT_ROOT, "valid_preprocessed"),
        help="Default directory to load plate images from.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class LPRNetTesterApp(tk.Tk):
    """Dark-themed Tkinter app for testing LPRNet on individual plate images."""

    def __init__(self, ocr_engine: EdgeLPRNet, default_dir: str):
        super().__init__()
        self.ocr_engine = ocr_engine
        self.default_dir = default_dir

        self.title("LPRNet Plate OCR Tester")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1060, 680)

        # State
        self.image_paths: list[str] = []
        self.current_idx = 0
        self._photo_orig = None
        self._photo_prep = None
        self._results: list[dict] = []  # cached results for batch

        self._build_ui()
        self._center_window(1140, 720)

        # Auto-load if directory exists
        if os.path.isdir(default_dir):
            self._load_directory(default_dir)

    # ── Window helpers ───────────────────────────────────────────────────────
    def _center_window(self, w, h):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    @staticmethod
    def _bind_hover(widget, hover_bg, normal_bg):
        widget.bind("<Enter>", lambda _: widget.config(bg=hover_bg))
        widget.bind("<Leave>", lambda _: widget.config(bg=normal_bg))

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ───────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_PANEL, pady=12)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="🔍  LPRNet Plate OCR Tester",
            font=FONT_TITLE, fg=TEXT_PRI, bg=BG_PANEL,
        ).pack()
        tk.Label(
            hdr, text="Load plate crop images · Preprocess · Recognise",
            font=FONT_SMALL, fg=TEXT_SEC, bg=BG_PANEL,
        ).pack()

        # ── Toolbar ──────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=BG_DARK, pady=8, padx=18)
        toolbar.pack(fill="x")

        btn_open_dir = tk.Button(
            toolbar, text="📂  Open Folder",
            font=("Segoe UI", 10, "bold"), fg="white", bg="#30363d",
            activebackground="#484f58", activeforeground="white",
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=self._on_open_folder,
        )
        btn_open_dir.pack(side="left", padx=(0, 6))
        self._bind_hover(btn_open_dir, "#484f58", "#30363d")

        btn_open_files = tk.Button(
            toolbar, text="🖼  Open Files",
            font=("Segoe UI", 10, "bold"), fg="white", bg="#30363d",
            activebackground="#484f58", activeforeground="white",
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=self._on_open_files,
        )
        btn_open_files.pack(side="left", padx=(0, 6))
        self._bind_hover(btn_open_files, "#484f58", "#30363d")

        self.btn_run_all = tk.Button(
            toolbar, text="▶  Run All",
            font=("Segoe UI", 10, "bold"), fg="white", bg=ACCENT,
            activebackground=ACCENT_HV, activeforeground="white",
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=self._on_run_all,
        )
        self.btn_run_all.pack(side="left", padx=(0, 6))
        self._bind_hover(self.btn_run_all, ACCENT_HV, ACCENT)

        # Navigation
        self.btn_prev = tk.Button(
            toolbar, text="◀  Prev",
            font=FONT_BODY, fg=TEXT_PRI, bg="#30363d",
            activebackground="#484f58", activeforeground="white",
            relief="flat", bd=0, padx=10, pady=8, cursor="hand2",
            command=self._on_prev,
        )
        self.btn_prev.pack(side="right", padx=(6, 0))
        self._bind_hover(self.btn_prev, "#484f58", "#30363d")

        self.btn_next = tk.Button(
            toolbar, text="Next  ▶",
            font=FONT_BODY, fg=TEXT_PRI, bg="#30363d",
            activebackground="#484f58", activeforeground="white",
            relief="flat", bd=0, padx=10, pady=8, cursor="hand2",
            command=self._on_next,
        )
        self.btn_next.pack(side="right", padx=(6, 0))
        self._bind_hover(self.btn_next, "#484f58", "#30363d")

        self.nav_label = tk.Label(
            toolbar, text="No images loaded",
            font=FONT_SMALL, fg=TEXT_SEC, bg=BG_DARK,
        )
        self.nav_label.pack(side="right", padx=(6, 0))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Body: split left/right ───────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DARK, padx=18, pady=12)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        # ── Left Column: image list ──────────────────────────────────────────
        left = tk.Frame(body, bg=BG_PANEL, bd=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(
            left, text="  Image Queue",
            font=FONT_HEAD, fg=TEXT_PRI, bg=BG_PANEL, anchor="w", pady=8,
        ).pack(fill="x")

        list_frame = tk.Frame(left, bg=BG_PANEL)
        list_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self.file_listbox = tk.Listbox(
            list_frame,
            bg=BG_INPUT, fg=CYAN, font=FONT_MONO,
            selectbackground=ACCENT, selectforeground="white",
            bd=0, relief="flat", highlightthickness=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        self.file_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # ── Right Column: preview + result ───────────────────────────────────
        right = tk.Frame(body, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)   # image previews get most space
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        # Image previews — stacked vertically, full width
        preview_frame = tk.Frame(right, bg=BG_DARK)
        preview_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=0)
        preview_frame.rowconfigure(1, weight=1)

        # Original image card ──────────────────────────────────────────────
        orig_card = tk.Frame(preview_frame, bg=BG_CARD, bd=0)
        orig_card.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5))

        orig_hdr = tk.Frame(orig_card, bg=BG_CARD)
        orig_hdr.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(
            orig_hdr, text="Original Crop",
            font=("Segoe UI", 10, "bold"), fg=CYAN, bg=BG_CARD, anchor="w",
        ).pack(side="left")
        self.lbl_orig_dim = tk.Label(
            orig_hdr, text="",
            font=FONT_SMALL, fg=TEXT_SEC, bg=BG_CARD, anchor="e",
        )
        self.lbl_orig_dim.pack(side="right")

        # Accent border frame around the image
        orig_border = tk.Frame(orig_card, bg=CYAN, bd=0)
        orig_border.pack(padx=8, pady=(4, 8), fill="both", expand=True)
        orig_inner = tk.Frame(orig_border, bg="#010409", bd=0)
        orig_inner.pack(padx=2, pady=2, fill="both", expand=True)
        self.lbl_orig = tk.Label(
            orig_inner, bg="#010409",
            text="No image", fg="#30363d", font=("Segoe UI", 11),
        )
        self.lbl_orig.pack(fill="both", expand=True, padx=4, pady=4)

        # Preprocessed image card ──────────────────────────────────────────
        prep_card = tk.Frame(preview_frame, bg=BG_CARD, bd=0)
        prep_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(5, 0))

        prep_hdr = tk.Frame(prep_card, bg=BG_CARD)
        prep_hdr.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(
            prep_hdr, text="Preprocessed (LPRNet Input)",
            font=("Segoe UI", 10, "bold"), fg=GOLD, bg=BG_CARD, anchor="w",
        ).pack(side="left")
        self.lbl_prep_dim = tk.Label(
            prep_hdr, text="",
            font=FONT_SMALL, fg=TEXT_SEC, bg=BG_CARD, anchor="e",
        )
        self.lbl_prep_dim.pack(side="right")

        # Accent border frame around the image
        prep_border = tk.Frame(prep_card, bg=GOLD, bd=0)
        prep_border.pack(padx=8, pady=(4, 8), fill="both", expand=True)
        prep_inner = tk.Frame(prep_border, bg="#010409", bd=0)
        prep_inner.pack(padx=2, pady=2, fill="both", expand=True)
        self.lbl_prep = tk.Label(
            prep_inner, bg="#010409",
            text="No image", fg="#30363d", font=("Segoe UI", 11),
        )
        self.lbl_prep.pack(fill="both", expand=True, padx=4, pady=4)

        # OCR Result card
        result_card = tk.Frame(right, bg=BG_CARD, bd=0, pady=16, padx=20)
        result_card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        tk.Label(
            result_card, text="OCR Result",
            font=FONT_HEAD, fg=TEXT_SEC, bg=BG_CARD, anchor="w",
        ).pack(fill="x")

        self.lbl_plate = tk.Label(
            result_card, text="—",
            font=FONT_PLATE, fg=PLATE_CLR, bg=BG_CARD, anchor="w",
        )
        self.lbl_plate.pack(fill="x", pady=(8, 4))

        # Details sub-row
        details = tk.Frame(result_card, bg=BG_CARD)
        details.pack(fill="x")

        self.lbl_valid = tk.Label(
            details, text="",
            font=FONT_BODY, fg=TEXT_SEC, bg=BG_CARD, anchor="w",
        )
        self.lbl_valid.pack(side="left")

        self.lbl_raw = tk.Label(
            details, text="",
            font=FONT_MONO, fg=TEXT_SEC, bg=BG_CARD, anchor="e",
        )
        self.lbl_raw.pack(side="right")

        self.lbl_filename = tk.Label(
            result_card, text="",
            font=FONT_SMALL, fg=TEXT_SEC, bg=BG_CARD, anchor="w", wraplength=500,
        )
        self.lbl_filename.pack(fill="x", pady=(8, 0))

        # ── Batch results log ────────────────────────────────────────────────
        log_frame = tk.Frame(right, bg=BG_CARD, bd=0)
        log_frame.grid(row=2, column=0, sticky="nsew")
        right.rowconfigure(2, weight=1)

        tk.Label(
            log_frame, text="  Batch Results Log",
            font=FONT_HEAD, fg=TEXT_PRI, bg=BG_CARD, anchor="w", pady=6,
        ).pack(fill="x")

        self.log_text = tk.Text(
            log_frame, height=8,
            bg=BG_INPUT, fg="#79c0ff", font=FONT_LOG,
            bd=0, relief="flat", insertbackground=TEXT_PRI,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # ── Footer status bar ────────────────────────────────────────────────
        footer = tk.Frame(self, bg=BG_PANEL, pady=6, padx=18)
        footer.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="Ready — load images to begin.")
        tk.Label(
            footer, textvariable=self.status_var,
            font=FONT_SMALL, fg=TEXT_SEC, bg=BG_PANEL, anchor="w",
        ).pack(fill="x")

    # ── File loading ─────────────────────────────────────────────────────────
    def _on_open_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder with plate crop images",
            initialdir=self.default_dir,
        )
        if folder:
            self._load_directory(folder)

    def _on_open_files(self):
        files = filedialog.askopenfilenames(
            title="Select plate crop images",
            initialdir=self.default_dir,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")],
        )
        if files:
            self.image_paths = list(files)
            self._results = [None] * len(self.image_paths)
            self.current_idx = 0
            self._populate_listbox()
            self._show_current()

    def _load_directory(self, folder: str):
        paths = []
        for f in sorted(os.listdir(folder)):
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTS:
                paths.append(os.path.join(folder, f))

        if not paths:
            messagebox.showinfo("No Images", f"No supported images found in:\n{folder}")
            return

        self.image_paths = paths
        self._results = [None] * len(paths)
        self.current_idx = 0
        self.status_var.set(f"Loaded {len(paths)} images from {os.path.basename(folder)}")
        self._populate_listbox()
        self._show_current()

    def _populate_listbox(self):
        self.file_listbox.delete(0, "end")
        for p in self.image_paths:
            self.file_listbox.insert("end", os.path.basename(p))
        if self.image_paths:
            self.file_listbox.selection_set(0)
        self._update_nav()

    # ── Navigation ───────────────────────────────────────────────────────────
    def _on_prev(self):
        if not self.image_paths:
            return
        self.current_idx = max(0, self.current_idx - 1)
        self.file_listbox.selection_clear(0, "end")
        self.file_listbox.selection_set(self.current_idx)
        self.file_listbox.see(self.current_idx)
        self._show_current()

    def _on_next(self):
        if not self.image_paths:
            return
        self.current_idx = min(len(self.image_paths) - 1, self.current_idx + 1)
        self.file_listbox.selection_clear(0, "end")
        self.file_listbox.selection_set(self.current_idx)
        self.file_listbox.see(self.current_idx)
        self._show_current()

    def _on_listbox_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            self.current_idx = sel[0]
            self._show_current()

    def _update_nav(self):
        total = len(self.image_paths)
        if total == 0:
            self.nav_label.config(text="No images loaded")
        else:
            self.nav_label.config(text=f"{self.current_idx + 1} / {total}")

    # ── Display current image + run OCR ──────────────────────────────────────
    def _show_current(self):
        if not self.image_paths:
            return

        self._update_nav()
        path = self.image_paths[self.current_idx]
        basename = os.path.basename(path)

        # Load original image
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self.lbl_plate.config(text="READ ERROR", fg=RED_ACC)
            self.lbl_filename.config(text=basename)
            self.status_var.set(f"Failed to read: {basename}")
            return

        # Display original (scaled up for visibility — plate crops are tiny)
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_pil = Image.fromarray(orig_rgb)
        oh, ow = img_bgr.shape[:2]
        self.lbl_orig_dim.config(text=f"{ow}×{oh} px")
        orig_display = self._fit_image(orig_pil, max_w=520, max_h=220, upscale=True)
        self._photo_orig = ImageTk.PhotoImage(orig_display)
        self.lbl_orig.config(image=self._photo_orig, text="")

        # Preprocess
        tensor_input = preprocess_plate(img_bgr)

        # Display preprocessed version
        prep_img = (tensor_input[0] * 255.0).astype(np.uint8)
        prep_pil = Image.fromarray(prep_img)
        ph, pw = prep_img.shape[:2]
        self.lbl_prep_dim.config(text=f"{pw}×{ph} px")
        prep_display = self._fit_image(prep_pil, max_w=520, max_h=220, upscale=True)
        self._photo_prep = ImageTk.PhotoImage(prep_display)
        self.lbl_prep.config(image=self._photo_prep, text="")

        # Run OCR
        ocr_dict = self.ocr_engine.extract_text(tensor_input)
        raw_text = ocr_dict["text"]
        result = format_ocr_result(raw_text)
        # Include confidence in result if needed, or we just keep the formatted result as is
        result["confidence"] = ocr_dict.get("confidence", 0.0)

        # Cache result
        self._results[self.current_idx] = result

        # Update result display
        plate = result["plate_number"]
        valid = result["valid"]

        self.lbl_plate.config(
            text=plate if plate else "—",
            fg=PLATE_CLR if valid else GOLD,
        )
        self.lbl_valid.config(
            text=f"{'✅ Valid PH Format' if valid else '⚠ Non-standard format'}",
            fg=PLATE_CLR if valid else GOLD,
        )
        self.lbl_raw.config(text=f"raw: \"{result['raw']}\"  |  conf: {result['confidence']:.3f}")
        self.lbl_filename.config(text=f"File: {basename}")

        self.status_var.set(f"Recognised: {plate}  ({'valid' if valid else 'non-standard'})  —  {basename}")

    @staticmethod
    def _fit_image(pil_img: Image.Image, max_w=520, max_h=220, upscale=False):
        """Scale image to fit within bounds, using LANCZOS for smooth upscaling."""
        w, h = pil_img.size
        if upscale or w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resampling = getattr(Image, "Resampling", Image)
            lanczos = getattr(resampling, "LANCZOS", getattr(Image, "LANCZOS", 1))
            return pil_img.resize((new_w, new_h), lanczos)
        return pil_img

    # ── Batch run ────────────────────────────────────────────────────────────
    def _on_run_all(self):
        if not self.image_paths:
            messagebox.showinfo("No Images", "Load images first.")
            return

        self.btn_run_all.config(state="disabled", text="⏳  Running…")
        self.status_var.set("Running batch OCR…")
        threading.Thread(target=self._run_all_worker, daemon=True).start()

    def _run_all_worker(self):
        total = len(self.image_paths)
        valid_count = 0
        log_lines = []

        for i, path in enumerate(self.image_paths):
            basename = os.path.basename(path)
            self.after(0, lambda idx=i: self.status_var.set(
                f"Processing {idx + 1}/{total}…"
            ))

            img_bgr = cv2.imread(path)
            if img_bgr is None:
                log_lines.append(f"  {basename:50s}  →  READ ERROR")
                self._results[i] = {"plate_number": "ERROR", "raw": "", "valid": False}
                continue

            tensor_input = preprocess_plate(img_bgr)
            raw_text = self.ocr_engine.extract_text(tensor_input)
            result = format_ocr_result(raw_text)
            self._results[i] = result

            plate = result["plate_number"]
            valid = result["valid"]
            if valid:
                valid_count += 1

            marker = "✓" if valid else "✗"
            log_lines.append(f"  {marker}  {basename:50s}  →  {plate}")

        # Summary
        summary = (
            f"\n{'═' * 60}\n"
            f"  BATCH COMPLETE: {total} images processed\n"
            f"  Valid PH plates: {valid_count}/{total}  "
            f"({valid_count / total * 100:.1f}%)\n"
            f"{'═' * 60}\n"
        )
        log_lines.append(summary)

        log_output = "\n".join(log_lines)

        def _update():
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", log_output)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
            self.btn_run_all.config(state="normal", text="▶  Run All")
            self.status_var.set(
                f"Batch done — {valid_count}/{total} valid PH plates "
                f"({valid_count / total * 100:.1f}%)"
            )
            # Highlight current selection
            if self.image_paths:
                self._show_current()

        self.after(0, _update)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("[INFO] Loading LPRNet TFLite model…")
    try:
        ocr_engine = EdgeLPRNet(model_path=args.model)
        print(f"[INFO] Model loaded: {args.model}")
    except Exception as e:
        print(f"[ERROR] Failed to load LPRNet model: {e}")
        sys.exit(1)

    app = LPRNetTesterApp(ocr_engine=ocr_engine, default_dir=args.dir)
    app.mainloop()


if __name__ == "__main__":
    main()
