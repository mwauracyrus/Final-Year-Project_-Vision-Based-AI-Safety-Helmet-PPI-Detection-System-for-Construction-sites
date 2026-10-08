import cv2
import numpy as np
import os
import time
import csv
import requests
import threading
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from flask import Flask, request

# =========================
# TELEGRAM CONFIG
# =========================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_telegram_alert(identity, violation, timestamp, image_path, gps_data):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    caption = (
        f"🚨 SAFETY ALERT 🚨\n"
        f"Name: {identity}\n"
        f"Violation: {violation}\n"
        f"Time: {timestamp}\n"
        f"GPS: {gps_data['lat']}, {gps_data['lng']}"
    )

    try:
        with open(image_path, "rb") as photo:
            files = {"photo": photo}
            data = {"chat_id": CHAT_ID, "caption": caption}

            r = requests.post(url, data=data, files=files, timeout=10)
            print("[TELEGRAM SENT]", r.json())

    except Exception as e:
        print("[TELEGRAM ERROR]", e)

# =========================================
# ROLE MAPPING
# =========================================
def get_role_from_helmet(color):
    if color == "WHITE":
        return "Engineer/Supervisor"
    elif color == "YELLOW":
        return "Worker"
    else:
        return "Unknown"
    
# =========================
# GPS SERVER (ESP RECEIVER)
# =========================
app_server = Flask(__name__)
latest_gps = {}

@app_server.route('/gps', methods=['GET'])
def gps_receiver():

    worker = request.args.get('worker')
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    time_val = request.args.get('time')

    latest_gps[worker] = {
        "lat": lat,
        "lng": lng,
        "time": time_val
    }

    print(f"[GPS RECEIVED] {worker}: {lat}, {lng}")

    return "OK"

def run_server():
    app_server.run(host="0.0.0.0", port=5000)

threading.Thread(target=run_server, daemon=True).start()

# =========================
# MODEL LOAD
# =========================
model = YOLO(r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt")

app = FaceAnalysis(name="buffalo_l", root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo")
app.prepare(ctx_id=0, det_size=(640, 640))

# =========================
# FACE DB
# =========================
face_db = np.load(
    r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy",
    allow_pickle=True
).item()

for k in face_db:
    face_db[k] = face_db[k] / np.linalg.norm(face_db[k])

# =========================
# LOGGING
# =========================
log_dir = "violations"
os.makedirs(log_dir, exist_ok=True)

csv_file = os.path.join(log_dir, "logs.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time", "Name", "Violation",
            "Helmet_Color", "Role",
            "Latitude", "Longitude", "Image"
        ])

# =========================
# CONTROL
# =========================
last_alert = {}
ALERT_DELAY = 10

# =========================
# FACE RECOGNITION
# =========================
def recognize_face(embedding):

    embedding = embedding / np.linalg.norm(embedding)

    best_name = "Unknown"
    best_dist = 999

    for name, db in face_db.items():
        dist = np.linalg.norm(embedding - db)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name if best_dist < 1.2 else "Unknown"

# =========================
# HELMET COLOR
# =========================
def detect_helmet_color(helmet_crop):
    hsv = cv2.cvtColor(helmet_crop, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))

    yellow = cv2.countNonZero(yellow_mask)
    white = cv2.countNonZero(white_mask)

    if yellow > white and yellow > 500:
        return "YELLOW"
    elif white > yellow and white > 500:
        return "WHITE"
    return "UNKNOWN"

# =========================
# SAVE VIOLATION
# =========================
def save_violation(frame, name, helmet_color):

    global last_alert

    now = time.time()

    if name in last_alert and now - last_alert[name] < ALERT_DELAY:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gps_data = latest_gps.get(name, {"lat":"UNKNOWN", "lng":"UNKNOWN"})

    filename = f"{name}_{timestamp.replace(':','-')}.jpg"
    path = os.path.join(log_dir, filename)

    cv2.imwrite(path, frame)

    role = "Engineer/Supervisor" if helmet_color == "WHITE" else "Worker"

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            name,
            "NO HELMET",
            helmet_color,
            role,
            gps_data["lat"],
            gps_data["lng"],
            path
        ])

    print("[VIOLATION SAVED]", name, gps_data)

    send_telegram_alert(name, "NO HELMET", timestamp, path, gps_data)

    last_alert[name] = now

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    helmet_boxes = []

    # HELMET DETECTION
    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if model.names[cls] == "class_0" and conf > 0.5:

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                color = detect_helmet_color(crop)

                helmet_boxes.append(((x1,y1,x2,y2), color))

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

    # FACE DETECTION
    faces = app.get(frame)

    for f in faces:

        x1,y1,x2,y2 = map(int, f.bbox)
        name = recognize_face(f.embedding)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        has_helmet = False
        helmet_color = "NONE"

        for (hx1,hy1,hx2,hy2), col in helmet_boxes:
            if hx1-100 < cx < hx2+100 and hy1-120 < cy < hy2+120:
                has_helmet = True
                helmet_color = col
                break

        if not has_helmet:
            save_violation(frame, name, helmet_color)
            color = (0,0,255)
            status = "NO HELMET"
        else:
            color = (0,255,0)
            status = "SAFE"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{name} | {status}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("FULL SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()