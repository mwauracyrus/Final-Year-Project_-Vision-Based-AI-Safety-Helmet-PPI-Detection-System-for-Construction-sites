import cv2
import numpy as np
import os
import time
import csv
import threading
import winsound
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from ultralytics import YOLO
from insightface.app import FaceAnalysis


# =====================================================
# PATHS
# =====================================================
BASE_DIR = r"F:/Documents/yolov5safetyhelmet-main (1)/Training5"

model_path = os.path.join(BASE_DIR, "best.pt")
face_db_path = os.path.join(BASE_DIR, "face_db.npy")
log_dir = os.path.join(BASE_DIR, "violations")
os.makedirs(log_dir, exist_ok=True)

csv_file = os.path.join(log_dir, "logs.csv")


# =====================================================
# LOAD MODELS
# =====================================================
model = YOLO(model_path)

face_app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))


# =====================================================
# FACE DB
# =====================================================
face_db = np.load(face_db_path, allow_pickle=True).item()

for k in face_db:
    face_db[k] = face_db[k] / np.linalg.norm(face_db[k])


# =====================================================
# CSV INIT
# =====================================================
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        csv.writer(f).writerow(["Time", "Name", "Status", "Image"])


# =====================================================
# FACE RECOGNITION
# =====================================================
def recognize_face(emb):
    emb = emb / np.linalg.norm(emb)

    best_name = "Unknown"
    best_score = -1

    for name, db_emb in face_db.items():
        db_emb = db_emb / np.linalg.norm(db_emb)
        score = np.dot(emb, db_emb)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < 0.35:
        return "Unknown"

    return best_name


# =====================================================
# HELMET COLOR
# =====================================================
def detect_helmet_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, (20,100,100), (35,255,255))
    white = cv2.inRange(hsv, (0,0,180), (180,50,255))

    y = cv2.countNonZero(yellow)
    w = cv2.countNonZero(white)

    if y > w and y > 500:
        return "YELLOW"
    elif w > y and w > 500:
        return "WHITE"
    else:
        return "UNKNOWN"


# =====================================================
# DASHBOARD
# =====================================================
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CCTV Safety Helmet System")
        self.resize(1300, 800)

        self.camera = QLabel()
        self.listbox = QListWidget()

        layout = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        left.addWidget(self.camera)

        right.addWidget(QLabel("Violations"))
        right.addWidget(self.listbox)

        layout.addLayout(left, 70)
        layout.addLayout(right, 30)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.last_alert = {}
        self.cooldown = 5


    # ================= SOUND =================
    def alert(self):
        threading.Thread(
            target=lambda: winsound.Beep(1200, 400),
            daemon=True
        ).start()


    # ================= SAVE =================
    def save_violation(self, frame, name):

        now = time.time()
        if now - self.last_alert.get(name, 0) < self.cooldown:
            return

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(log_dir, f"{name}_{ts}.jpg")

        cv2.imwrite(path, frame)

        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow([ts, name, "NO HELMET", path])

        self.listbox.insertItem(0, f"{name} | NO HELMET | {ts}")

        self.alert()

        self.last_alert[name] = now


    # ================= MAIN LOOP =================
    def update(self):

        ret, frame = self.cap.read()
        if not ret:
            return

        results = model(frame)

        helmets = []

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if label == "class_0" and conf > 0.5:

                    crop = frame[y1:y2, x1:x2]
                    color = detect_helmet_color(crop)

                    helmets.append((x1,y1,x2,y2))

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

        faces = face_app.get(frame)

        for f in faces:

            x1,y1,x2,y2 = map(int, f.bbox)
            name = recognize_face(f.embedding)

            cx, cy = (x1+x2)//2, (y1+y2)//2

            has_helmet = False

            for h in helmets:
                hx1,hy1,hx2,hy2 = h

                if hx1-100 < cx < hx2+100 and hy1-120 < cy < hy2+120:
                    has_helmet = True

            if has_helmet:
                color = (0,255,0)
                status = "SAFE"
            else:
                color = (0,0,255)
                status = "NO HELMET"
                self.save_violation(frame, name)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{name} | {status}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        img = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        self.camera.setPixmap(QPixmap.fromImage(img))


# =====================================================
# RUN APP
# =====================================================
app = QApplication([])
win = Dashboard()
win.show()
app.exec_()
