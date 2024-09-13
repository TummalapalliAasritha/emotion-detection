import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

face_cascade_path = r'C:\Users\DELL\OneDrive\Desktop\emotionminiproject\haarcascade_frontalface_default.xml'
emotion_model_path = r"C:\Users\DELL\OneDrive\Desktop\emotionminiproject\emotion_detection_model_100epochs.h5\emotion_detection_model_100epochs.h5"

if not os.path.exists(face_cascade_path):
    raise FileNotFoundError(f"The file {face_cascade_path} does not exist. Please check the path.")
face_classifier = cv2.CascadeClassifier(face_cascade_path)


if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"The file {emotion_model_path} does not exist. Please check the path.")
emotion_model = load_model(emotion_model_path)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)  

def detect_emotion(emotion_model, face_img):
    face_img = face_img.astype('float32') / 255.0  
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    emotion_preds = emotion_model.predict(face_img)[0]
    print("Emotion predictions shape:", emotion_preds.shape)
    print("Emotion predictions:", emotion_preds)
    emotion_index = emotion_preds.argmax()
    emotion_confidence = emotion_preds[emotion_index]
    emotion_label = class_labels[emotion_index]
    return emotion_label, emotion_confidence
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        try:
            emotion_label, emotion_confidence = detect_emotion(emotion_model, roi_gray)
            print(f"Emotion: {emotion_label} - Confidence: {emotion_confidence:.2f}")
            label_position = (x, y)
            cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Error in emotion detection:", e)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
