# 🚗 Parking YOLO — License Plate & Vehicle Detection

A real-time parking lot object detection web application powered by **YOLOv8** and **Flask**. It streams live video from a webcam or video file and detects vehicles and license plates using a custom-trained YOLO model.

---

## 📸 Features

- 🔴 **Live Video Streaming** via webcam or video file through a browser interface
- 🧠 **Custom-Trained YOLO Model** detecting 4 classes:
  - `car`
  - `large vehicle`
  - `motorcycle`
  - `plate` (license plate)
- 🌐 **Flask Web Server** serving a real-time MJPEG video stream
- 📦 Lightweight, easy to set up and run locally

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) | Object detection model |
| [OpenCV](https://opencv.org/) | Video capture & frame processing |
| [Flask](https://flask.palletsprojects.com/) | Web server & video streaming |
| [Jinja2](https://jinja.palletsprojects.com/) | HTML templating |
| Python 3.8+ | Core language |

---

## 📁 Project Structure

```
parking_yolo/
├── app.py                  # Flask application & video streaming logic
├── model.pt                # Custom-trained YOLO model weights
├── parking_yolo.ipynb      # Jupyter Notebook for model training & evaluation
├── templates/
│   └── index.html          # Web UI for the live video stream
├── venv/                   # Python virtual environment (not tracked)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/parking_yolo.git
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
pip install flask ultralytics opencv-python
```

### 4. Add Your Model Weights

Place your custom-trained YOLO model file (`best.pt` or `model.pt`) in the root project directory.

> **Note:** The `app.py` currently loads `best.pt`. Rename your file or update the path in `app.py`:
> ```python
> model = YOLO('best.pt')
> ```

---

## 🚀 Running the Application

```bash
python app.py
```

The server will start on **http://0.0.0.0:5000**. Open your browser and navigate to:

```
http://localhost:5000
```

You should see the live video stream with bounding boxes drawn around detected vehicles and license plates.

---

## 🎥 Video Source Configuration

By default, the app uses your **webcam (device index `0`)**. To use a video file instead, update `app.py`:

```python
# Webcam
cap = cv2.VideoCapture(0)

# Video file
cap = cv2.VideoCapture('parking_video.mp4')
```

---

## 🧪 Model Training

The `parking_yolo.ipynb` notebook documents the full model training pipeline, including:

- Dataset preparation and annotation
- YOLOv8 model training configuration
- Validation and evaluation metrics
- Exporting the trained model weights

---

## 📊 Detection Classes

| Class | Description |
|---|---|
| `car` | Standard passenger vehicles |
| `large vehicle` | Trucks, buses, vans |
| `motorcycle` | Motorcycles and scooters |
| `plate` | Vehicle license plates |

---

## 📝 License

This project is intended for educational and research purposes.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [Roboflow](https://roboflow.com/) for dataset management tools
- [OpenCV](https://opencv.org/) for video processing
