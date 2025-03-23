import os
import face_recognition
import pickle

def train_faces():
    known_encodings = []
    known_names = []
    
    for person in os.listdir("dataset"):
        person_dir = os.path.join("dataset", person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_encodings.append(encoding[0])
                    known_names.append(person)
    
    # Save encodings
    with open("models/encodings.pkl", "wb") as f:
        pickle.dump((known_encodings, known_names), f)
    print(f"Trained {len(known_names)} faces!")

if __name__ == "__main__":
    train_faces()