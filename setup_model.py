import os
import subprocess
from ultralytics import YOLO

def setup_yolo_model():
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("database", exist_ok=True)

    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics package...")
        subprocess.check_call(["pip", "install", "ultralytics"])
    
    
    try:
        print("Downloading YOLOv8 face detection model...")
        model = YOLO('yolov8n-face.pt')
        model.save("models/yolov8n-face.pt")
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("You can manually download it from:")
        print("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")

if __name__ == "__main__":
    setup_yolo_model()
    print("Setup complete! Model saved in: models/yolov8n-face.pt")