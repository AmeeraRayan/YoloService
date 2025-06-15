import boto3
import json
import requests
import time
import os

# Load from env
QUEUE_URL = os.environ.get("SQS_QUEUE_URL")
REGION_NAME = "eu-north-1"
YOLO_URL = "http://localhost:8000/predict"  # YOLO runs locally inside EC2

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

                # Call local YOLO FastAPI service
                try:
                    resp = requests.post(YOLO_URL, json=body)
                    resp.raise_for_status()
                    print("✅ YOLO processed the image:", resp.json())
                except Exception as e:
                    print("❌ Error calling YOLO:", e)

                # Delete from SQS
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
                print("🗑️ Message deleted from SQS")

        except Exception as e:
            print("❌ Consumer error:", e)
            time.sleep(5)

if __name__ == "__main__":
    consume_messages()