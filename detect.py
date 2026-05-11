import cv2
import numpy as np
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import img_to_array
import pygame
import os

# Initialize pygame mixer for audio
pygame.mixer.init()
ALARM_SOUND_PATH = "alarm.wav"

if os.path.exists(ALARM_SOUND_PATH):
    alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)
else:
    print(f"Warning: {ALARM_SOUND_PATH} not found. Audio alert will not work.")
    alarm_sound = None

def play_alarm():
    if alarm_sound and not pygame.mixer.get_busy():
        alarm_sound.play()

def stop_alarm():
    if alarm_sound and pygame.mixer.get_busy():
        pygame.mixer.stop()

def detect_drowsiness(model_path="drowsiness_model.h5", N_CONSECUTIVE_FRAMES=15):
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Please train the model first.")
        return

    print("Loading model...")
    model = load_model(model_path)
    
    # Load Haar Cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    print("Starting webcam...")
    cap = cv2.VideoCapture(0)

    closed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Invert the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # We'll just look at the largest face for simplicity, or all faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            
            eye_status = "Open"
            confidence = 0.0
            
            # Predict state for each detected eye
            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Preprocess eye for CNN
                eye_img = cv2.resize(eye_img, (64, 64))
                eye_img = img_to_array(eye_img)
                eye_img = np.expand_dims(eye_img, axis=0)
                eye_img = eye_img / 255.0
                
                # Prediction
                # Assuming class 0 is Closed_Eyes, class 1 is Open_Eyes
                pred = model.predict(eye_img, verbose=0)
                closed_prob = pred[0][0]
                open_prob = pred[0][1]
                
                if closed_prob > open_prob:
                    eye_status = "Closed"
                    confidence = closed_prob
                    break # If one eye is closed, we consider checking it. Alternatively, check if both are closed.
                else:
                    eye_status = "Open"
                    confidence = open_prob
                    
            # Logic for drowsiness
            if eye_status == "Closed":
                closed_frames += 1
            else:
                closed_frames = 0
                stop_alarm()
                
            # Trigger alarm
            if closed_frames >= N_CONSECUTIVE_FRAMES:
                play_alarm()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Overlay status
            label = f"{eye_status} ({confidence:.2f})"
            color = (0, 255, 0) if eye_status == "Open" else (0, 0, 255)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.imshow("Drowsiness Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    detect_drowsiness()
