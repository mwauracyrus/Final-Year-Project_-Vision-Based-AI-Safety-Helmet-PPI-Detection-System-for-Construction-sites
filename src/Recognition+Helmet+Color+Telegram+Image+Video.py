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

def send_telegram_alert(identity, violation, timestamp, image_path):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    caption = (
        f"🚨 SAFETY ALERT 🚨\n"
        f"Name: {identity}\n"
        f"Violation: {violation}\n"
        f"Time: {timestamp}"
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
def get_role(color):
    if color == "WHITE":
        return "Engineer/Supervisor"
    elif color == "YELLOW":
        return "Worker"
    return "Unknown"


# =========================================
# LOAD MODELS
# =========================================
model = YOLO(r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt")

app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)
app.prepare(ctx_id=0, det_size=(640, 640))


# =========================================
# FACE DB
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
        writer.writerow(["Time", "Name", "Violation", "Helmet", "Role", "Image"])


# =========================================
# ANTI-SPAM CONTROL
# =========================================
last_alert = {}
COOLDOWN = 10  # seconds


# =========================================
# FACE RECOGNITION
# =========================================
def recognize(face_embedding):

    face_embedding = face_embedding / np.linalg.norm(face_embedding)

    best_name = "Unknown"
    best_dist = 999

    for name, db_emb in face_db.items():
        dist = np.linalg.norm(face_embedding - db_emb)

        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name if best_dist < 1.2 else "Unknown"


# =========================================
# HELMET COLOR DETECTION
# =========================================
def helmet_color_detect(crop):

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, (20,100,100), (35,255,255))
    white  = cv2.inRange(hsv, (0,0,180), (180,50,255))

    y = cv2.countNonZero(yellow)
    w = cv2.countNonZero(white)

    if y > w and y > 500:
        return "YELLOW"
    elif w > y and w > 500:
        return "WHITE"
    return "UNKNOWN"


# =========================================
# SAVE VIOLATION
# =========================================
def save_violation(frame, name, helmet):

    now = time.time()

    if name in last_alert and now - last_alert[name] < COOLDOWN:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = f"{name}_{timestamp.replace(':','-')}.jpg"
    path = os.path.join(log_dir, filename)

    cv2.imwrite(path, frame)

    role = get_role(helmet)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, "NO HELMET", helmet, role, path])

    print("[VIOLATION SAVED]", name)

    send_telegram_alert(name, "NO HELMET", timestamp, path)

    last_alert[name] = now


# =========================================
# VIDEO INPUT (CHANGE THIS)
# =========================================
VIDEO_PATH = r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/helmet.mp4"   # 👈 change here

cap = cv2.VideoCapture(VIDEO_PATH)


# =========================================
# MAIN LOOP
# =========================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    helmet_boxes = []

    # ----------------------------
    # HELMET DETECTION
    # ----------------------------
    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if model.names[cls] == "class_0" and conf > 0.5:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                color = helmet_color_detect(crop)

                helmet_boxes.append({"box": (x1,y1,x2,y2), "color": color})

                role = get_role(color)

                if color == "WHITE":
                    col = (255,255,255)
                elif color == "YELLOW":
                    col = (0,255,255)
                else:
                    col = (0,255,0)

                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, f"{color} | {role}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # ----------------------------
    # FACE DETECTION
    # ----------------------------
    faces = app.get(frame)

    for f in faces:

        x1,y1,x2,y2 = map(int, f.bbox)
        name = recognize(f.embedding)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        has_helmet = False
        helmet_color = "NONE"

        for h in helmet_boxes:
            hx1,hy1,hx2,hy2 = h["box"]

            if hx1-100 < cx < hx2+100 and hy1-120 < cy < hy2+120:
                has_helmet = True
                helmet_color = h["color"]
                break

        if has_helmet:
            status = "SAFE"
            col = (0,255,0)
        else:
            status = "NO HELMET"
            col = (0,0,255)

            save_violation(frame, name, helmet_color)

        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        cv2.putText(frame, f"{name} | {status}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    cv2.imshow("VIDEO TEST - SAFETY SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()