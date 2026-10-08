import cv2
import numpy as np
import os
import time
import csv
import requests
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# =========================================
# TELEGRAM CONFIG
# =========================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# =========================================
# GPS DATA (FROM HELMET)
# =========================================
gps_data = {
    "worker": "Unknown",
    "lat": "Unknown",
    "lng": "Unknown",
    "time": "Unknown"
}


# =========================================
# TELEGRAM ALERT (ONLY VIOLATION)
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
            r = requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=15
            )

        print("📤 Telegram:", r.status_code, r.text)

    except Exception as e:
        print("❌ Telegram error:", e)


# =========================================
# ROLE MAPPING
# =========================================
def get_role(color):
    if color == "WHITE":
        return "Engineer/Supervisor"
    elif color == "YELLOW":
        return "Worker"
    return "Unknown"


def helmet_label(color):
    if color == "WHITE":
        return "White Helmet: Engineer/Supervisor"
    elif color == "YELLOW":
        return "Yellow Helmet: Worker"
    return "Unknown Helmet"


# =========================================
# LOAD MODELS
# =========================================
model = YOLO(r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt")

face_app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)
face_app.prepare(ctx_id=0, det_size=(640,640))


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
# LOGGING
# =========================================
log_dir = "violations"
os.makedirs(log_dir, exist_ok=True)

csv_file = os.path.join(log_dir, "logs.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time","Name","Status","Helmet","Role","Image"
        ])


# =========================================
# ANTI-SPAM
# =========================================
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
# HELMET COLOR
# =========================================
def detect_color(crop):

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    y = cv2.inRange(hsv, (20,100,100), (35,255,255))
    w = cv2.inRange(hsv, (0,0,180), (180,50,255))

    yellow = cv2.countNonZero(y)
    white = cv2.countNonZero(w)

    if yellow > white and yellow > 500:
        return "YELLOW"
    elif white > yellow and white > 500:
        return "WHITE"

    return "UNKNOWN"


# =========================================
# SAVE VIOLATION
# =========================================
def save_violation(frame, name, helmet_color):

    now = time.time()

    if name in last_alert and now - last_alert[name] < ALERT_DELAY:
        return

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path = os.path.join(log_dir, f"{name}_{ts}.jpg")
    cv2.imwrite(path, frame)

    print("🚨 VIOLATION:", name)

    send_violation_alert(name, ts, path)

    last_alert[name] = now


# =========================================
# VIDEO INPUT (CHANGE THIS)
# =========================================
video_path = "F:/Documents/yolov5safetyhelmet-main (1)/Training5/helmet.mp4"   # <-- PUT YOUR VIDEO HERE
cap = cv2.VideoCapture(video_path)


# =========================================
# MAIN LOOP
# =========================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    helmets = []

    # DETECT HELMETS
    for r in results:
        for box in r.boxes:

            if float(box.conf[0]) < 0.5:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            color = detect_color(crop)
            helmets.append((x1,y1,x2,y2,color))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.putText(frame,helmet_label(color),
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(255,255,0),2)


    # FACE DETECTION
    faces = face_app.get(frame)

    for f in faces:

        x1,y1,x2,y2 = map(int, f.bbox)
        name = recognize(f.embedding)

        cx,cy = (x1+x2)//2,(y1+y2)//2

        has_helmet = False
        helmet_color = "NONE"

        for (hx1,hy1,hx2,hy2,c) in helmets:
            if hx1-100 < cx < hx2+100 and hy1-120 < cy < hy2+120:
                has_helmet = True
                helmet_color = c
                break

        if has_helmet:

            status = "SAFE"

            # GPS DISPLAY ONLY (NO ALERT)
            gps_text = f"GPS: {gps_data['lat']}, {gps_data['lng']}"

            cv2.putText(frame, gps_text,
                        (x1,y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,255),2)

        else:

            status = "NO HELMET"
            save_violation(frame, name, helmet_color)

        cv2.rectangle(frame,(x1,y1),(x2,y2),
                      (0,255,0) if has_helmet else (0,0,255),2)

        cv2.putText(frame,f"{name} | {status}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0) if has_helmet else (0,0,255),2)


    cv2.imshow("VIDEO SAFETY SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()