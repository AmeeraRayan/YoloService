import boto3
import os
from datetime import datetime
from storage.base import Storage

class DynamoDBStorage(Storage):
    def init(self):
        self.table_name = os.getenv("DYNAMODB_TABLE", "Predictions")
        self.region = os.getenv("AWS_REGION", "eu-north-1")
        self.dynamodb = boto3.resource("dynamodb", region_name=self.region)
        self.table = self.dynamodb.Table(self.table_name)

    def save_prediction(self, uid, original_path, predicted_path):
        item = {
            "uid": uid,
            "original_path": original_path,
            "predicted_path": predicted_path,
            "created_at": datetime.utcnow().isoformat(),
            "detections": []  # initialize empty list to add detections later
        }
        self.table.put_item(Item=item)
        print(f"[DynamoDB] Saved prediction for UID: {uid}")

    def save_detection(self, uid, label, score, bbox):
        # Convert bbox to string or JSON-serializable format
        detection = {
            "label": label,
            "score": score,
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