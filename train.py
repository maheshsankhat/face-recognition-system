import cv2
import os
import time
import numpy as np 



person_name = input("Enter Person Name : ")



dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

os.makedirs(person_path, exist_ok=True)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



cap = cv2.VideoCapture(0)

count = 0

print("Starting Face Capture...")

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # FACE CROP
        face = gray[y:y+h, x:x+w]

        # RESIZE FACE
        face = cv2.resize(face, (200, 200))

        count += 1

        # SAVE IMAGE
        img_path = os.path.join(person_path, f"{count}.jpg")

        cv2.imwrite(img_path, face)

        # DRAW RECTANGLE
        cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 255, 0),2)

        # SHOW COUNT
        cv2.putText(frame,f"Photo : {count}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

        cv2.imshow("Face Capture", frame)

        print(f"Photo {count} Saved")

    
        # WAIT 2 SECONDS
     

        if count % 5 == 0:
            print("Change Position...")
            time.sleep(2)
            
       
        
        

    cv2.imshow("Face Capture", frame)

    # TOTAL PHOTOS
    if count >= 50:
        break

    # PRESS ESC TO EXIT
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Face Dataset Collection Complete!")



# trainig a recognization model 

faces =[]
labels=[]
label_map ={}


current_label =0

for person_name  in os.listdir(dataset_path):
    
    person_path = os.path.join(dataset_path,person_name)
    
    if not os.path.isdir(person_path):
        continue
    
    label_map[current_label] = person_name
    
    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (200, 200))

        faces.append(img)

        labels.append(current_label)

    current_label += 1

# CREATE RECOGNIZER
recognizer = cv2.face.LBPHFaceRecognizer_create()


recognizer.train(faces, np.array(labels))

recognizer.save("trained_model.yml")

# SAVE LABELS
np.save("labels.npy", label_map)

print("Training Complete!")