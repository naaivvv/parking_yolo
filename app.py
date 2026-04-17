import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load your custom-trained YOLO26 model
model = YOLO('best.pt') 

# Define the 4 classes from your dataset
class_names = ['car', 'large vehicle', 'motorcycle', 'plate']

def generate_frames():
    # Initialize the video capture object. 
    # Use 0 for webcam. If you have a video file, replace 0 with the file path (e.g., 'parking_video.mp4')
    cap = cv2.VideoCapture(0) 
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            # Run YOLO inference on the frame
            # A confidence threshold of 0.25 matches your validation script
            results = model.predict(source=frame, conf=0.25)
            
            # Plot the bounding boxes onto the frame
            annotated_frame = results[0].plot()
            
            # Encode the frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format for the web stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Pass the classes to the frontend
    return render_template('index.html', classes=class_names)

@app.route('/video_feed')
def video_feed():
    # Stream the video generator to the webpage
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Start the Flask development server
    app.run(host='0.0.0.0', port=5000, debug=True)