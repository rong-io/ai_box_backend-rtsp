#!/bin/bash

WORKDIR="/opt/nanoowl"
IMAGE="nanoowl"
ENGINE_PATH="data/owl_image_encoder_patch32.engine"

echo "Running hardware_monitoring.py..."
python3 "$(dirname "$0")/hardware_monitoring.py" &
MONITOR_PID=$!

echo "Running jetson-containers..."
jetson-containers run --workdir "$WORKDIR" "$(autotag $IMAGE)" bash -c "python3 /app/demo.py $ENGINE_PATH" &
CONTAINER_PID=$!

wait $MONITOR_PID
wait $CONTAINER_PID

echo "successed."
