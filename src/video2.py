import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("F:/Documents/yolov5safetyhelmet-main (1)/yolov5safetyhelmet-main/best.pt")

# Open the video
cap = cv2.VideoCapture("F:/Documents/yolov5safetyhelmet-main (1)/Training5/helmet.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame if needed
    frame_resized = cv2.resize(frame, (1020, 600))

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model(frame_rgb, conf=0.25)  # confidence threshold

    # Draw boxes on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Helmet Detection", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
