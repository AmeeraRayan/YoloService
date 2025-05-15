import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import boto3
import traceback
import logging
import torch

torch.cuda.is_available = lambda: False

app = FastAPI()

# Constants
BUCKET_NAME = "ameera-polybot-images"
REGION_NAME = "eu-north-1"
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

# Logging
logging.basicConfig(level=logging.INFO)

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Load model
model = YOLO("yolov8n.pt")

# Initialize DB
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()

# UID from image name
def generate_uid(image_name):
    base_name = image_name.split('.')[0]
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    return str(uuid.uuid5(namespace, base_name))

# S3
def download_from_s3(image_name, local_path):
    logging.info(f"Downloading {image_name} from S3...")
    s3 = boto3.client("s3", region_name=REGION_NAME)
    s3.download_file(BUCKET_NAME, image_name, local_path)

def upload_to_s3(file_path, s3_key):
    logging.info(f"Uploading {file_path} to S3 key: {s3_key}")
    s3 = boto3.client("s3", region_name=REGION_NAME)
    s3.upload_file(file_path, BUCKET_NAME, s3_key)

# DB saves
def save_prediction_session(uid, original_image, predicted_image):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.get("/")
def read_root():
    return {"message": "Welcome!"}



@app.post("/predict")
async def predict_s3(request: Request):
    data = await request.json()
    image_name = data.get("image_name")

    # שלב 1: בדיקת שדות חובה
    if not image_name:
        print("[ERROR] Missing image_name in request")
        raise HTTPException(status_code=400, detail="Missing image_name")

    try:
        print(f"[INFO] Starting prediction for image: {image_name}")

        # יצירת UID ושמות קבצים
        uid = str(uuid.uuid4())
        ext = os.path.splitext(image_name)[1]
        base_name = os.path.splitext(image_name)[0]
        original_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        predicted_path = os.path.join(PREDICTED_DIR, f"{uid}_predicted{ext}")

        # שלב 2: הורדת התמונה מ־S3
        print("[INFO] Downloading image from S3...")
        download_from_s3(image_name, original_path)

        # שלב 3: הרצת YOLO
        print("[INFO] Running YOLO model...")
        results = model(original_path, device="cpu")
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        # שלב 4: שמירת המידע במסד נתונים
        print("[INFO] Saving prediction to database...")
        save_prediction_session(uid, original_path, predicted_path)

        detected_labels = []
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            save_detection_object(uid, label, score, bbox)
            detected_labels.append(label)

        # שלב 5: העלאה חזרה ל־S3
        print("[INFO] Uploading predicted image to S3...")
        predicted_s3_key = f"predicted/{uid}_predicted{ext}"
        upload_to_s3(predicted_path, predicted_s3_key)

        print("[INFO] Prediction completed successfully.")
        return {
            "prediction_uid": uid,
            "detection_count": len(detected_labels),
            "labels": detected_labels,
            "predicted_s3_key": predicted_s3_key
        }

    except Exception as e:
        print(f"[ERROR] Failed to process image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.1"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

