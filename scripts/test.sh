#!/bin/bash

# Start the server in the background
./target/debug/oml &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for the server to start
sleep 2 

# Send a training request
TRAIN_START=$(date +%s)
curl -s -X POST "http://localhost:8080/training" -H "Content-Type: application/json" -d '1.1' &
TRAIN_PID=$!

# Send concurrent inference requests
INF1_START=$(date +%s)
curl -s -X POST "http://localhost:8080/inference" -H "Content-Type: application/json" -d '2.0' &
INF1_PID=$!

sleep 6

INF2_START=$(date +%s)
curl -s -X POST "http://localhost:8080/inference" -H "Content-Type: application/json" -d '2.0' &
INF2_PID=$!

# Wait for all requests to complete
wait $TRAIN_PID
TRAIN_END=$(date +%s)

wait $INF1_PID
INF1_END=$(date +%s)

wait $INF2_PID
INF2_END=$(date +%s)

# Kill the server
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null  # Suppress the "Terminated" message

# Compute durations and overlaps
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
INF1_DURATION=$((INF1_END - INF1_START))
INF2_DURATION=$((INF2_END - INF2_START))

echo "Training Duration: $TRAIN_DURATION ms"
echo "Inference 1 Duration: $INF1_DURATION ms"
echo "Inference 2 Duration: $INF2_DURATION ms"

# Check if inference requests started after training began, but finished before training ended
if [[ $INF1_START -ge $TRAIN_START && $INF1_END -le $TRAIN_END ]] && \
   [[ $INF2_START -ge $TRAIN_START && $INF2_END -le $TRAIN_END ]]; then
    echo "Test Passed: Inference requests ran concurrently during training."
else
    echo "Test Failed: No concurrency detected between training and inference requests."
fi
