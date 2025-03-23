import cv2
import os
from ultralytics import YOLO

def enroll_face():
    model = YOLO('models/yolov8n-face.pt')
    name = input("Enter person's name: ").strip().replace(" ", "_")
    output_dir = os.path.join("dataset", name)
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)  
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        results = model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            count += 1
            cv2.imwrite(f"{output_dir}/{count}.jpg", face)  # Save face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Enroll Face", frame)
        if cv2.waitKey(1) == ord('q') or count >= 20:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"{name} enrolled with {count} images!")

if __name__ == "__main__":
    enroll_face()