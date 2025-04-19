import cv2
import numpy as np
import os

# Load input video
input_path = "video.mp4"
output_path = "output_tracked.mp4"

# Open input video
cap = cv2.VideoCapture(input_path)

# Set forced resolution
frame_width = 1920
frame_height = 1080
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Kalman Filter class
class TrackedObject:
    def __init__(self, x, y):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.predicted = (x, y)
        self.trace = []

    def update(self, cx, cy):
        measured = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measured)
        prediction = self.kalman.predict()
        self.predicted = (int(prediction[0]), int(prediction[1]))
        self.trace.append(self.predicted)
        if len(self.trace) > 30:
            self.trace.pop(0)

# List of tracked objects
tracked_objects = []

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 1920x1080 explicitly
    frame = cv2.resize(frame, (frame_width, frame_height))

    fgmask = fgbg.apply(frame)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            detections.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for det in detections:
        matched = False
        for obj in tracked_objects:
            px, py = obj.predicted
            if abs(det[0] - px) < 50 and abs(det[1] - py) < 50:
                obj.update(*det)
                matched = True
                break
        if not matched:
            tracked_objects.append(TrackedObject(*det))

    # Predict and draw
    for obj in tracked_objects:
        obj.kalman.predict()
        cv2.circle(frame, obj.predicted, 5, (255, 0, 0), -1)
        for i in range(1, len(obj.trace)):
            cv2.line(frame, obj.trace[i - 1], obj.trace[i], (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Optional preview
    cv2.imshow("Kalman Multi-Object Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print("Tracking complete. Output saved as:", output_path)
