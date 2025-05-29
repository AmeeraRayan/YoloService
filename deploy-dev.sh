#!/bin/bash
# Copy the service file for dev
sudo cp yolo-dev.service /etc/systemd/system/

# Reload daemon and restart the service
sudo systemctl daemon-reload
sudo systemctl restart yolo-dev.service
sudo systemctl enable yolo-dev.service

# Check if the service is active
if ! systemctl is-active --quiet yolo-dev.service; then
  echo "❌ yolo-dev.service is not running."
  sudo systemctl status yolo-dev.service --no-pager
  exit 1
fi

echo '✅ Yolo dev service is running successfully!'