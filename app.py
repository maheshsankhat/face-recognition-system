import streamlit as st
import os
import cv2
import time
import numpy as np

# PAGE CONFIG
st.set_page_config(page_title="Face Recognition System", layout="centered")

st.title("Face Recognition Attendance System")

# SIDEBAR
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go To",
    [
        "📁 Upload Dataset",
        "🧠 Train Model",
        "📊 Face Recognition"
    ]
)

# COMMON PATHS
dataset_path = "dataset"

# ==============================
# DATASET PAGE
# ==============================

if page == "📁 Upload Dataset":

    st.header("Upload Face Dataset")

    person_name = st.text_input("Enter Person Name")

    if st.button("Start Face Capture"):

        if person_name == "":
            st.warning("Please enter a name")
            st.stop()

        person_path = os.path.join(dataset_path, person_name)

        os.makedirs(person_path, exist_ok=True)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(0)

        stframe = st.empty()

        count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("Camera not working")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                face = gray[y:y+h, x:x+w]

                face = cv2.resize(face, (200, 200))

                count += 1

                img_path = os.path.join(person_path, f"{count}.jpg")

                cv2.imwrite(img_path, face)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Photo: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                if count % 5 == 0:
                    time.sleep(0.5)

            stframe.image(frame, channels="BGR")

            if count >= 50:
                break

        cap.release()

        st.success("Face Dataset Collection Complete!")

# ==============================
# TRAIN MODEL PAGE
# ==============================

elif page == "🧠 Train Model":

    st.header("Train Face Recognition Model")

    if st.button("Train Model"):

        faces = []
        labels = []
        label_map = {}

        current_label = 0

        for person_name in os.listdir(dataset_path):

            person_path = os.path.join(dataset_path, person_name)

            if not os.path.isdir(person_path):
                continue

            label_map[current_label] = person_name

            for image_name in os.listdir(person_path):

                img_path = os.path.join(person_path, image_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = cv2.resize(img, (200, 200))

                faces.append(img)

                labels.append(current_label)

            current_label += 1

        if len(faces) == 0:
            st.error("No dataset found")
            st.stop()

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.train(faces, np.array(labels))

        recognizer.save("trained_model.yml")

        np.save("labels.npy", label_map)

        st.success("Training Complete!")

# ==============================
# FACE RECOGNITION PAGE
# ==============================

elif page == "📊 Face Recognition":

    st.header("Real-Time Face Recognition")

    if not os.path.exists("trained_model.yml"):
        st.error("Train model first")
        st.stop()

    if st.button("Start Recognition"):

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read("trained_model.yml")

        label_map = np.load(
            "labels.npy",
            allow_pickle=True
        ).item()

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        cap = cv2.VideoCapture(0)

        stframe = st.empty()

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("Camera not working")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                face_img = gray[y:y+h, x:x+w]

                face_img = cv2.resize(face_img, (200, 200))

                label, confidence = recognizer.predict(face_img)

                if confidence < 50:

                    name = label_map[label]

                    text = name

                    color = (0, 255, 0)

                else:

                    text = "Unknown"

                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

            stframe.image(frame, channels="BGR")

        cap.release()