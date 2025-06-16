import boto3
import os
from datetime import datetime
from storage.base import Storage
import json
from decimal import Decimal
class DynamoDBStorage(Storage):
    def __init__(self):
        self.table_name = os.getenv("DYNAMODB_TABLE", "Predictions")
        self.region = os.getenv("AWS_REGION", "eu-north-1")
        self.dynamodb = boto3.resource("dynamodb", region_name=self.region)
        self.table = self.dynamodb.Table(self.table_name)

    def save_prediction(self, uid, original_path, predicted_path, chat_id=None):
        item = {
            "uid": uid,
            "original_path": original_path,
            "predicted_path": predicted_path,
            "created_at": datetime.utcnow().isoformat(),
            "detections": [] , # initialize empty list to add detections later
            "chat_id": chat_id
        }
        self.table.put_item(Item=item)
        print(f"[DynamoDB] Saved prediction for UID: {uid}")

    def save_detection(self, uid, label, score, bbox):
        # Convert bbox to string or JSON-serializable format
        detection = {
            "label": label,
            "score": Decimal(str(score)),
            "bbox": bbox
        }

        # Use update expression to append detection to the list
        self.table.update_item(
            Key={"uid": uid},
            UpdateExpression="SET detections = list_append(if_not_exists(detections, :empty_list), :d)",
            ExpressionAttributeValues={
                ":d": [detection],
                ":empty_list": []
            }
        )
        print(f"[DynamoDB] Added detection for UID: {uid} -> {label} ({score:.2f})")

    def get_prediction(self, uid):
        try:
            response = self.table.get_item(Key={'uid': uid})
            item = response.get('Item')
            if not item:
                return None
            return {
                "prediction_uid": item.get("uid"),
                "original_image": item.get("original_image"),
                "predicted_image": item.get("predicted_image"),
                "labels": json.loads(item.get("labels", "[]")),
                "score": float(item.get("score", 0)),
                "box": json.loads(item.get("box", "[]")),
                "timestamp": item.get("timestamp"),
                "chat_id":item.get("chat_id")
            }
        except Exception as e:
            print(f"[ERROR] get_prediction failed: {e}")
            return None

    def get_predictions_by_score(self, min_score: float):
        min_score_decimal = Decimal(str(min_score))
        try:
            # Scan all items from the table
            response = self.table.scan()
            items = response.get("Items", [])

            matched_predictions = []

            for item in items:
                detections = item.get("detections", [])
                for detection in detections:
                    score = Decimal(str(detection.get("score", 0)))
                    if score >= min_score_decimal:
                        matched_predictions.append({
                            "uid": item["uid"],
                            "timestamp": item.get("created_at") or item.get("timestamp")
                        })
                        break  # avoid adding same UID twice

            return matched_predictions

        except Exception as e:
            print(f"[ERROR] get_predictions_by_score failed: {e}")
            return []