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
import threading
import json
import time
import requests
torch.cuda.is_available = lambda: False
from storage.base import Storage
from storage.sqlite_storage import SQLiteStorage
# from storage.dynamodb_storage import DynamoDBStorage  # for future
app = FastAPI()

# Select storage backend
storage_type = os.getenv("STORAGE_TYPE", "sqlite")
if storage_type == "dynamodb":
    from storage.dynamodb_storage import DynamoDBStorage
    storage = DynamoDBStorage()
else:
    storage = SQLiteStorage()
print("[INFO] Using DynamoDBStorage" if storage_type == "dynamodb" else "[INFO] Using SQLiteStorage")
# Constants
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
# Logging
logging.basicConfig(level=logging.INFO)

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Load model
model = YOLO("yolov8n.pt")

# UID from image name
def generate_uid(image_name):
    base_name = image_name.split('.')[0]
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    return str(uuid.uuid5(namespace, base_name))

# S3
def download_from_s3(image_name, local_path ,bucket_name ,region_name):
    logging.info(f"Downloading {image_name} from S3...")
    s3 = boto3.client("s3", region_name=region_name)
    s3.download_file(bucket_name, image_name, local_path)

def upload_to_s3(file_path, s3_key,bucket_name,region_name):
    logging.info(f"Uploading {file_path} to S3 key: {s3_key}")
    s3 = boto3.client("s3", region_name=region_name)
    s3.upload_file(file_path, bucket_name, s3_key)

@app.get("/")
def read_root():
    return {"message": "Welcome!"}

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    return storage.get_predictions_by_score(min_score)

@app.get("/predictions/{uid}")
def get_prediction(uid: str):
    result = storage.get_prediction(uid)
    if result:
        return result
    raise HTTPException(status_code=404, detail="Prediction not found")


@app.post("/predict")
async def predict_s3(request: Request):
    data = await request.json()
    image_name = data.get("image_name")
    bucket_name = data.get("bucket_name")
    region_name = data.get("region_name")

    # שלב 1: בדיקת שדות חובה
    if not image_name or not bucket_name or not region_name:
        print("[ERROR] Missing image_name in request")
        raise HTTPException(status_code=400, detail="Missing image_name")

    try:
        print(f"[INFO] Starting prediction for image: {image_name}")

        # יצירת UID ושמות קבצים
        uid = str(uuid.uuid4())
        print(f"[DEBUG] Generated UID: {uid}")
        base_name , ext= os.path.splitext(os.path.basename(image_name))
        original_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        predicted_name=f"{base_name}_predicted{ext}"
        predicted_path = os.path.join(PREDICTED_DIR, predicted_name)

        # שלב 2: הורדת התמונה מ־S3
        print("[INFO] Downloading image from S3...")
        download_from_s3(image_name, original_path, bucket_name,region_name)

        # שלב 3: הרצת YOLO
        print("[INFO] Running YOLO model...")
        results = model(original_path, device="cpu")
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        # שלב 4: שמירת המידע במסד נתונים
        print("[INFO] Saving prediction to database...")
        storage.save_prediction(uid, original_path, predicted_path)

        detected_labels = []
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            storage.save_detection(uid, label, score, bbox)
            detected_labels.append(label)

        print("🧠 Detected objects:", ", ".join(detected_labels))  #
        # שלב 5: העלאה חזרה ל־S3
        print("[INFO] Uploading predicted image to S3...")
        predicted_s3_key = f"predicted/{predicted_name}"
        upload_to_s3(predicted_path, predicted_s3_key , bucket_name,region_name)

        print("[INFO] Prediction completed successfully.")
        return {
            "prediction_uid": uid,
            "original_image":image_name,
            "predicted_image":predicted_name,
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

# 👇 Load these only once
QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
YOLO_URL = "http://localhost:8000/predict"  # Since it's local to container
REGION_NAME = "eu-north-1"

def consume_messages():
    sqs = boto3.client("sqs", region_name=REGION_NAME)
    print(f"🟢 Listening to queue: {QUEUE_URL}")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=5,
                WaitTimeSeconds=10
            )
            messages = response.get("Messages", [])
            if not messages:
                print("🟡 No messages, waiting...")
                time.sleep(1)
                continue

            for msg in messages:
                body = json.loads(msg["Body"])
                print(f"📥 Message received: {body}")

                try:
                    resp = requests.post(YOLO_URL, json=body)
                    resp.raise_for_status()
                    print("✅ YOLO processed the image:", resp.json())
                except Exception as e:
                    print("❌ Error calling YOLO:", e)

                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
                print("🗑️ Message deleted from SQS")

        except Exception as e:
            print("❌ Consumer error:", e)
            time.sleep(5)

# Start the background consumer when FastAPI starts
@app.on_event("startup")
def start_consumer_thread():
    print("🚀 Starting SQS consumer in background thread...")
    t = threading.Thread(target=consume_messages, daemon=True)
    t.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

