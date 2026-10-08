import cv2
import time
from ultralytics import YOLO

# =========================================
# CONFIGURATION
# =========================================
# For Webcam: Set to 0
# For Video: Set to your filename string (e.g., "test_video.mp4")
SOURCE = 0

# Load ONLY the YOLO object detection model
model = YOLO(r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt")

# =========================================
# ROLE MAPPING LOGIC ONLY
# =========================================
def get_role(color):
    if color == "WHITE": 
        return "Engineer/Supervisor"
    if color == "YELLOW": 
        return "Worker"
    return "Unknown"

def helmet_label(color):
    if color == "WHITE": 
        return "White Helmet: Supervisor"
    if color == "YELLOW": 
        return "Yellow Helmet: Worker"
    return "No Helmet / Visitor"

# =========================================
# SMART ADAPTIVE COLOR DETECTION LOGIC
# =========================================
def detect_color(crop):
    # Convert cropped helmet box to HSV color space
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # 1. Isolate Yellow mask (Hue 15-40 captures all shades of yellow/gold even in shadows)
    yellow_mask = cv2.inRange(hsv, (15, 60, 80), (40, 255, 255))
    
    # 2. Isolate White mask (Low saturation 'S' ensures yellow reflections are ignored)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 45, 255))

    yellow_pixels = cv2.countNonZero(yellow_mask)
    white_pixels = cv2.countNonZero(white_mask)

    # Smart ratio gating to prevent reflections/glare from converting yellow to white
    if yellow_pixels > 300 and yellow_pixels > (white_pixels * 0.4):
        return "YELLOW"
    elif white_pixels > 400 and white_pixels > yellow_pixels:
        return "WHITE"
        
    return "UNKNOWN"

# =========================================
# STREAM PROCESSING LOOP
# =========================================
cap = cv2.VideoCapture(SOURCE)

# Handle dynamic delay sizing if a video path is supplied instead of a live feed
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / video_fps) if (video_fps and isinstance(SOURCE, str)) else 1

print("Starting Optimized Role Mapping Test Stream... Press 'ESC' to exit.")

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("End of stream or source file not found.")
        break

    # Run YOLO object detection
    results = model(frame, imgsz=320, verbose=False)

    for r in results:
        for box in r.boxes:
            # Confidence filter
            if float(box.conf[0]) < 0.5:
                continue

            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # 1. Determine Helmet Color via smart ratios
            color = detect_color(crop)
            
            # 2. Map Color to Professional Role Text
            display_text = helmet_label(color)
            role_logging = get_role(color)

            # Assign drawing properties based on color status
            if color == "YELLOW":
                box_color = (0, 255, 255)   # Yellow box
            elif color == "WHITE":
                box_color = (255, 255, 255) # White box
            else:
                box_color = (0, 165, 255)   # Orange box for Unknown

            # Print out matching assignments dynamically to terminal console
            print(f"Detected: {color} | Assigned Role: {role_logging}")

            # 3. Draw UI visuals onto the display frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.imshow("ROLE MAPPING ONLY ISOLATION TEST", frame)

    # Maintain normal speed tracking if testing on an external video path
    processing_time = int((time.time() - start_time) * 1000)
    dynamic_wait = max(1, frame_delay - processing_time) if isinstance(SOURCE, str) else 1

    # Press Esc key to exit out of the window loop
    if cv2.waitKey(dynamic_wait) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()