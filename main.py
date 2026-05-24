import cv2
import os
import numpy as np
import csv
from datetime import datetime


# LOAD FACE RECOGNIZER
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trained_model.yml")


# LOAD LABELS

label_map = np.load("labels.npy", allow_pickle=True).item()


# LOAD FACE DETECTOR


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



cap = cv2.VideoCapture(0)


marked = set()


# FACE DETECTION LOOP


while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # FACE CROP
        face_img = gray[y:y+h, x:x+w]

        # RESIZE
        face_img = cv2.resize(face_img, (200, 200))

        # PREDICT
        label, confidence = recognizer.predict(face_img)

        
        # RECOGNITION
       

        if confidence < 60:

            name = label_map[label]

            text = f"{name} ({round(confidence,2)})"

            color = (0, 255, 0)

            

        else:

            text = "Unknown"

            color = (0, 0, 255)

        
        # DRAW RECTANGLE
       

        cv2.rectangle(frame,(x, y),(x+w, y+h),color, 2)

        
        # SHOW NAME
        

        cv2.putText(frame,text,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    
    # SHOW WINDOW
   

    cv2.imshow("Face Attendance System", frame)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLOSE
# =========================

cap.release()

cv2.destroyAllWindows()