# 🚗 Parking YOLO — License Plate & Vehicle Detection

A local ALPR (Automatic License Plate Recognition) system powered by **YOLOv8** and a custom **LPRNet TFLite** model. It detects vehicles and license plates in a single YOLO pass, then reads the plate characters using an on-device TFLite model — no cloud API required.

---

## 📸 Features

- 🧠 **Two-Stage AI Pipeline** — YOLOv8 for detection, LPRNet TFLite for OCR
- 🚗 **4-Class Detection** — `car`, `large vehicle`, `motorcycle`, `plate`
- 🖥️ **Desktop GUI** (`src/main.py`) — Tkinter UI with a **Capture Frame** button; grabs a single webcam shot on demand and displays annotated results
- 🖼️ **Static Image Test Script** (`test_image.py`) — run the full pipeline on any image file from the command line
- 🇵🇭 **Philippine Plate Optimised** — charset and validation tuned for PH plates (e.g. `ABC1234`)
- 📦 **100% Local** — all inference runs on-device, no internet needed

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) | Vehicle & plate detection |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | On-device LPRNet OCR inference |
| [OpenCV](https://opencv.org/) | Frame capture & image processing |
| [Tkinter](https://docs.python.org/3/library/tkinter.html) | Desktop GUI |
| [Pillow](https://python-pillow.org/) | Image rendering inside Tkinter |
| Python 3.8+ | Core language |

---

## 📁 Project Structure

```
parking_yolo/
├── src/
│   ├── main.py             # Tkinter desktop UI (Capture Frame button)
│   ├── detect.py           # VehiclePlateDetector — YOLOv8 wrapper
│   ├── ocr.py              # EdgeLPRNet — TFLite OCR engine
│   ├── preprocess.py       # Plate crop preprocessing for LPRNet input
│   └── utils.py            # Drawing helpers & OCR output formatter
├── models/
│   ├── yolo26_custom.pt    # Custom-trained YOLOv8 weights
│   ├── ph001.tflite        # LPRNet TFLite — Philippine plates
│   ├── ccpd001.tflite      # LPRNet TFLite — CCPD dataset (alt)
│   └── recognition.tflite  # General recognition model (alt)
├── templates/
│   └── index.html          # Legacy Flask web UI
├── test_image.py           # CLI test script for static images
├── app.py                  # Legacy Flask streaming app
├── parking_yolo.ipynb      # Model training & evaluation notebook
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/naaivvv/parking_yolo.git
cd parking_yolo
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Model Files

Place the following files inside the `models/` directory:

| File | Description |
|---|---|
| `yolo26_custom.pt` | Custom YOLOv8 weights for vehicle & plate detection |
| `ph001.tflite` | LPRNet TFLite model (Philippine plates) |
| `ccpd001.tflite` | LPRNet TFLite model (CCPD dataset, alternative) |
| `recognition.tflite` | General recognition TFLite model (alternative) |

---

## 🚀 Running the Desktop App

Launch the Tkinter GUI from the project root:

```bash
python src/main.py
```

A window will open. Click **📷 Capture Frame** to:
1. Open the webcam and grab a single frame
2. Run YOLO detection on all objects
3. Run LPRNet OCR on every detected plate crop
4. Display the annotated image and plate results in the UI

> The webcam is only open for the duration of the capture — there is no continuous video stream.

---

## 🖼️ Test Script — Static Image

Use `test_image.py` to run the full pipeline on an image file without launching the GUI.

### Basic Usage

```bash
# From the project root
venv\Scripts\python.exe test_image.py path\to\your\image.jpg
```

### YOLO Only (Skip OCR)

```bash
venv\Scripts\python.exe test_image.py path\to\image.jpg --no-ocr
```

### Custom Model Paths

```bash
venv\Scripts\python.exe test_image.py path\to\image.jpg --yolo models\yolo26_custom.pt --lpr models\ph001.tflite
```

### Headless (No Display Window)

```bash
# Saves annotated result to test_output.jpg without opening a window
venv\Scripts\python.exe test_image.py path\to\image.jpg --no-show
```

### All Options

| Flag | Default | Description |
|---|---|---|
| `image` | *(required)* | Path to the input image |
| `--yolo` | `models/yolo26_custom.pt` | Path to YOLO `.pt` model |
| `--lpr` | `models/recognition.tflite` | Path to TFLite LPRNet model |
| `--conf` | `0.25` | YOLO confidence threshold |
| `--no-ocr` | — | Skip OCR, show YOLO detections only |
| `--output` | `test_output.jpg` | Path to save the annotated output image |
| `--no-show` | — | Don't open a display window |

---

## 📊 Detection Classes

| Class | Description |
|---|---|
| `car` | Standard passenger vehicles |
| `large vehicle` | Trucks, buses, vans |
| `motorcycle` | Motorcycles and scooters |
| `plate` | Vehicle license plates |

---

## 🧪 Model Training

The `parking_yolo.ipynb` notebook documents the full model training pipeline, including:

- Dataset preparation and annotation
- YOLOv8 model training configuration
- Validation and evaluation metrics
- Exporting the trained model weights

---

## 📝 License

This project is intended for educational and research purposes.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [TensorFlow Lite](https://www.tensorflow.org/lite) for on-device OCR inference
- [Roboflow](https://roboflow.com/) for dataset management tools
- [OpenCV](https://opencv.org/) for image and video processing
