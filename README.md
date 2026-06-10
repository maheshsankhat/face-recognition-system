# Face Recognition Attendance System

## Overview

A real-time Face Recognition System built using Python, OpenCV, Tkinter, and InsightFace ArcFace embeddings.

The system detects faces from a webcam feed, recognizes registered users, and displays their identities with confidence scores in real time.

## Features

* Real-time webcam face recognition
* ArcFace-based facial embeddings
* Unknown face detection
* Face tracking and smoothing
* Tkinter GUI
* Offline operation
* Multi-person recognition
* Confidence score display
* Face database management

## Technologies Used

* Python 3.10
* OpenCV
* InsightFace
* ONNX Runtime
* NumPy
* Pillow
* Tkinter

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

### Create Environment

```bash
conda create -n faceapp python=3.10
conda activate faceapp
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python main.py
```

## Build Executable

```bash
pyinstaller --noconfirm --onedir --windowed main.py
```

## Project Workflow

1. Capture webcam frame
2. Detect face using InsightFace detector
3. Generate ArcFace embeddings
4. Compare embeddings with stored database
5. Recognize user
6. Display confidence score
7. Track face across frames

## Model

The project uses the InsightFace Buffalo_L model:

* Face Detection
* Face Recognition
* Landmark Detection

## Performance

* Real-time recognition
* CPU supported
* Offline operation
* Multi-face detection

## Future Improvements

* Attendance logging
* Database management panel
* Export attendance to Excel
* REST API integration
* Cloud deployment

## Author

Mahesh Sankhat
