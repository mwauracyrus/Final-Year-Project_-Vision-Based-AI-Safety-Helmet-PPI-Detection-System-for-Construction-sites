import cv2
import numpy as np
import os
import time
import csv
import requests
import threading
from flask import Flask, request
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# =========================================
# TELEGRAM CONFIG
# =========================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# =========================================
# GPS SERVER (FROM HELMET ONLY)
# =========================================
gps_data = {
    "worker": "Unknown",
    "lat": "Unknown",
    "lng": "Unknown",
    "time": "Unknown"
}

app = Flask(__name__)

@app.route("/gps")
def gps():
    global gps_data
    gps_data["worker"] = request.args.get("worker", "Unknown")
    gps_data["lat"] = request.args.get("lat", "Unknown")
    gps_data["lng"] = request.args.get("lng", "Unknown")
    gps_data["time"] = request.args.get("time", "Unknown")
    return "OK"

def run_gps():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_gps, daemon=True).start()


# =========================================
# TELEGRAM ALERT
# =========================================
def send_violation_alert(identity, timestamp, image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    caption = (
        f"🚨 SAFETY VIOLATION 🚨\n"
        f"Name: {identity}\n"
        f"Violation: NO HELMET\n"
        f"Time: {timestamp}\n"
    )
    try:
        with open(image_path, "rb") as photo:
            requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=5  # Fast timeout so it doesn't block
            )
    except Exception as e:
        print("❌ Telegram Error:", e)


# =========================================
# ROLE MAPPING
# =========================================
def get_role(color):
    if color == "WHITE": return "Engineer/Supervisor"
    if color == "YELLOW": return "Worker"
    return "Unknown"

def helmet_label(color):
    if color == "WHITE": return "White Helmet: Supervisor"
    if color == "YELLOW": return "Yellow Helmet: Worker"
    return "Unknown Helmet"


# =========================================
# LOAD MODELS (FAST SETTINGS)
# =========================================
model = YOLO(r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt")

face_app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)
# Reduced detection size to 320x320 for lightning-fast face processing
face_app.prepare(ctx_id=0, det_size=(320, 320))


# =========================================
# FACE DATABASE
# =========================================
face_db = np.load(
    r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy",
    allow_pickle=True
).item()

for k in face_db:
    face_db[k] = face_db[k] / np.linalg.norm(face_db[k])


# =========================================
# LOGGING & ANTI-SPAM
# =========================================
log_dir = "violations"
os.makedirs(log_dir, exist_ok=True)
csv_file = os.path.join(log_dir, "logs.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time","Name","Status","Helmet","Role","Image"])

last_alert = {}
ALERT_DELAY = 10


# =========================================
# FACE RECOGNITION
# =========================================
def recognize(emb):
    emb = emb / np.linalg.norm(emb)
    best = "Unknown"
    best_d = 999

    for name, db in face_db.items():
        d = np.linalg.norm(emb - db)
        if d < best_d:
            best_d = d
            best = name

    return best if best_d < 1.2 else "Unknown"


# =========================================
# INTEGRATED ADAPTIVE COLOR DETECTION
# =========================================
def detect_color(crop):
    # Convert cropped helmet box to HSV color space
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # 1. Isolate Yellow mask (captures yellow/gold even under deep shadows)
    yellow_mask = cv2.inRange(hsv, (15, 60, 80), (40, 255, 255))
    
    # 2. Isolate White mask (strict low saturation prevents mistaking yellow glare for white)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 45, 255))

    yellow_pixels = cv2.countNonZero(yellow_mask)
    white_pixels = cv2.countNonZero(white_mask)

    # Use smart density verification ratios instead of absolute pixel counts
    if yellow_pixels > 300 and yellow_pixels > (white_pixels * 0.4): 
        return "YELLOW"
    elif white_pixels > 400 and white_pixels > yellow_pixels: 
        return "WHITE"
    return "UNKNOWN"


def save_violation(frame, name, helmet_color):
    now = time.time()
    if name in last_alert and now - last_alert[name] < ALERT_DELAY:
        return

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(log_dir, f"{name}_{ts}.jpg")
    cv2.imwrite(path, frame)

    role = get_role(helmet_color)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, name, "NO HELMET", helmet_color, role, path])

    # Threaded alert dispatch so backend network requests won't lag the video stream
    threading.Thread(target=send_violation_alert, args=(name, ts, path), daemon=True).start()
    last_alert[name] = now


# =========================================
# HIGH-SPEED CAMERA PROCESSING LOOP
# =========================================
cap = cv2.VideoCapture(0)

# Set buffer size to 1 so we always process the freshest live frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
cached_helmets = []
cached_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🚀 Run AI models every 3 frames to drastically save processing power
    if frame_count % 3 == 0 or not cached_faces:
        
        # 1. Run YOLO at a lower internal resolution (imgsz=320) for speed
        results = model(frame, imgsz=320, verbose=False)
        cached_helmets = []

        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                color = detect_color(crop)
                cached_helmets.append((x1, y1, x2, y2, color))

        # 2. Run Face Detection
        cached_faces = face_app.get(frame)

    # --- DRAWING & LOGIC LAYER (Runs every frame using cached tracks) ---
    
    # Draw Helmets
    for (hx1, hy1, hx2, hy2, color) in cached_helmets:
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
        cv2.putText(frame, helmet_label(color), (hx1, hy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Process Faces
    for f in cached_faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        name = recognize(f.embedding)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        has_helmet = False
        helmet_color = "NONE"

        for (hx1, hy1, hx2, hy2, c) in cached_helmets:
            if hx1 - 100 < cx < hx2 + 100 and hy1 - 120 < cy < hy2 + 120:
                has_helmet = True
                helmet_color = c
                break

        if has_helmet:
            status = "SAFE"
            color_theme = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_theme, 2)
            cv2.putText(frame, f"{name} | {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_theme, 2)
            
            # Show live GPS only when helmeted
            gps_text = f"GPS: {gps_data['lat']}, {gps_data['lng']}"
            cv2.putText(frame, gps_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            status = "NO HELMET"
            color_theme = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_theme, 2)
            cv2.putText(frame, f"{name} | {status}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_theme, 2)
            
            save_violation(frame, name, helmet_color)

    cv2.imshow("FINAL SAFETY SYSTEM (CLEAN LOGIC)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
