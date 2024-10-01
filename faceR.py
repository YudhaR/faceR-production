import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict

# Inisialisasi model YOLOv8
model = YOLO(r'D:\Project\faceR-production\yolov8n-face.pt')  # Ganti dengan path yang sesuai untuk model Anda
model.conf = 0.20  # Threshold confidence
model.maxdet = 10  # Maksimal deteksi

# Buka kamera
cap = cv2.VideoCapture(0)  # 0 untuk kamera default

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Check if results contain boxes
        if results[0].boxes is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for the latest frames
                    track.pop(0)

                # Draw the tracking lines
                # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(
                #     annotated_frame,
                #     [points],
                #     isClosed=False,
                #     color=(230, 230, 230),
                #     thickness=2,  # Adjust thickness as needed
                # )
        else:
            # No detections found
            annotated_frame = frame  # Show the original frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the camera stream is not available
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
