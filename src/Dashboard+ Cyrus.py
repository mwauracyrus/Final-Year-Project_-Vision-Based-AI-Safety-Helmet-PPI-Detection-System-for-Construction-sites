import sys
import cv2
import numpy as np
import os
import time
import csv
import requests
import threading
import pyttsx3   # NEW (replaces winsound)

from datetime import datetime
from flask import Flask, request

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem
)

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


# ====================================================
# TELEGRAM
# ====================================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ====================================================
# GPS SERVER
# ====================================================
gps_data = {
    "worker": "Unknown",
    "lat": "Unknown",
    "lng": "Unknown",
    "time": ""
}

app_server = Flask(__name__)

@app_server.route("/gps")
def receive_gps():

    global gps_data

    gps_data["worker"] = request.args.get("worker", "Unknown")
    gps_data["lat"] = request.args.get("lat", "Unknown")
    gps_data["lng"] = request.args.get("lng", "Unknown")
    gps_data["time"] = request.args.get("time", "")

    print("[GPS RECEIVED]", gps_data)
    return "OK"


def run_server():
    app_server.run(host="0.0.0.0", port=5000)


threading.Thread(target=run_server, daemon=True).start()


# ====================================================
# LOAD MODELS
# ====================================================
model = YOLO(r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt")

face_app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)

face_app.prepare(ctx_id=0, det_size=(640,640))


# ====================================================
# FACE DB
# ====================================================
face_db = np.load(
    r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy",
    allow_pickle=True
).item()

for k in face_db:
    face_db[k] = face_db[k] / np.linalg.norm(face_db[k])


# ====================================================
# TTS ENGINE (NEW)
# ====================================================
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)

def speak_warning(name):
    try:
        text = f"{name}, wear your helmet"
        tts_engine.say(text)
        tts_engine.runAndWait()
    except:
        pass


# ====================================================
# CONTINUOUS ALERT TRACKER (NEW)
# ====================================================
active_violations = {}   # key: name -> last spoken time


# ====================================================
# LOGGING
# ====================================================
log_dir = r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/violations"
os.makedirs(log_dir, exist_ok=True)

csv_file = os.path.join(log_dir, "logs.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time","Name","Violation","Helmet","Role","Latitude","Longitude","Image"
        ])


# ====================================================
# FUNCTIONS
# ====================================================
def send_telegram_alert(identity, timestamp, image_path):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    caption = (
        f"🚨 SAFETY ALERT 🚨\n\n"
        f"Name:{identity}\n"
        f"Violation:NO HELMET\n"
        f"Time:{timestamp}\n"
        f"Worker:{gps_data['worker']}\n"
        f"Lat:{gps_data['lat']}\n"
        f"Lng:{gps_data['lng']}"
    )

    try:
        with open(image_path, "rb") as photo:
            requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": photo}
            )
    except Exception as e:
        print("[TELEGRAM ERROR]", e)


# ====================================================
# CONTINUOUS ALARM SYSTEM (UPDATED)
# ====================================================
def play_alarm(identity):

    now = time.time()

    # initialize dict
    if identity not in active_violations:
        active_violations[identity] = 0

    # repeat every 4 seconds
    if now - active_violations[identity] >= 4:
        speak_warning(identity)
        active_violations[identity] = now


def stop_alarm(identity):
    if identity in active_violations:
        del active_violations[identity]


# ====================================================
# OTHER FUNCTIONS
# ====================================================
def get_role_from_helmet(color):

    if color == "WHITE":
        return "Engineer/Supervisor"
    elif color == "YELLOW":
        return "Worker"
    return "Unknown"


def recognize_face(embedding):

    embedding = embedding / np.linalg.norm(embedding)

    best = "Unknown"
    min_dist = 999

    for name, db in face_db.items():
        dist = np.linalg.norm(embedding - db)
        if dist < min_dist:
            min_dist = dist
            best = name

    return best if min_dist < 1.2 else "Unknown"


def detect_helmet_color(crop):

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, np.array([20,100,100]), np.array([35,255,255]))
    white = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,50,255]))

    y = cv2.countNonZero(yellow)
    w = cv2.countNonZero(white)

    if y > w and y > 500:
        return "YELLOW"
    elif w > y and w > 500:
        return "WHITE"

    return "UNKNOWN"


# ====================================================
# DASHBOARD
# ====================================================
class Dashboard(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("AI SAFETY DASHBOARD")
        self.resize(1400, 800)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Time","Event","Status"])

        layout = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(self.video_label)

        right = QVBoxLayout()
        right.addWidget(self.table)

        layout.addLayout(left, 70)
        layout.addLayout(right, 30)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)


    def add_log(self, event, status):

        row = self.table.rowCount()
        self.table.insertRow(row)

        now = datetime.now().strftime("%H:%M:%S")

        self.table.setItem(row, 0, QTableWidgetItem(now))
        self.table.setItem(row, 1, QTableWidgetItem(event))
        self.table.setItem(row, 2, QTableWidgetItem(status))


    def save_violation(self, frame, identity, helmet):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file = f"{identity}_{timestamp}.jpg"
        path = os.path.join(log_dir, file)

        cv2.imwrite(path, frame)

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                identity,
                "NO HELMET",
                helmet,
                "Unknown",
                gps_data["lat"],
                gps_data["lng"],
                path
            ])

        send_telegram_alert(identity, timestamp, path)

        self.add_log(identity, "NO HELMET")


    def update_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            return

        results = model(frame)

        helmets = []

        # DETECT HELMETS
        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "class_0" and conf > 0.5:

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    helmet = detect_helmet_color(crop)
                    helmets.append(((x1,y1,x2,y2), helmet))


        faces = face_app.get(frame)

        for face in faces:

            x1,y1,x2,y2 = map(int, face.bbox)
            identity = recognize_face(face.embedding)

            cx = (x1 + x2)//2
            cy = (y1 + y2)//2

            found = False
            helmet = "NONE"

            for (hx1,hy1,hx2,hy2), col in helmets:

                if hx1-100 < cx < hx2+100 and hy1-120 < cy < hy2+120:
                    found = True
                    helmet = col
                    break

            if found:

                status = "SAFE"
                color = (0,255,0)

                # STOP VOICE WHEN SAFE
                stop_alarm(identity)

            else:

                status = "NO HELMET"
                color = (0,0,255)

                self.save_violation(frame, identity, helmet)

                # CONTINUOUS VOICE ALERT
                play_alarm(identity)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.putText(
                frame,
                f"{identity}|{status}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h,w,ch = frame.shape

        img = QImage(frame.data, w, h, ch*w, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(img))


    def closeEvent(self, event):
        self.cap.release()
        event.accept()


# ====================================================
# RUN
# ====================================================
app_gui = QApplication(sys.argv)

window = Dashboard()
window.show()

sys.exit(app_gui.exec_())
