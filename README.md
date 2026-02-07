# CanineVet – YOLOv11 Dog Posture Analysis

CanineVet is a Flask web application that uses YOLOv11 pose estimation to analyze canine musculoskeletal posture (hips, elbows, and key joints) from a single dog image. It is developed as a thesis project for automated screening of hip and elbow health.

## Features

- Upload dog image (side/front view) via a modern Tailwind UI
- YOLOv11 pose model trained on Stanford Dog-Pose (24 keypoints)
- Visual overlay of colored keypoints on the dog image
- Simple hip/elbow health scores and status ("healthy" / "monitor")
- User registration, login, and saving past assessments with SQLite

## Requirements

- Python 3.10+ (recommended)
- pip

## Setup

```bash
git clone https://github.com/<BaekYeowoon>/caninevet-thesis.git
cd caninevet-thesis

python -m venv venv
venv\Scripts\activate   # on Windows
# source venv/bin/activate   # on macOS/Linux

pip install -r requirements.txt

Download trained YOLO weights
Download the trained YOLOv11 dog pose model (best.pt) from this Google Drive folder:

https://drive.google.com/drive/folders/1z62LchsLEvr8b47cf-gR1jBe3xS9hRHi?usp=drive_link

After downloading best.pt, update the TRAINED_WEIGHTS path in app.py to point to where you saved it. For example:

python
TRAINED_WEIGHTS = r"C:\Users\yourname\Models\dog_pose\best.pt"
If TRAINED_WEIGHTS is not valid or the file is missing, the app will fall back to anatomical keypoints or you can re-train using the training endpoint.

Running the App
bash
python app.py
Then open:

http://127.0.0.1:5000

in your browser to access the CanineVet UI