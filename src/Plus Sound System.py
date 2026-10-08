import cv2
import numpy as np
import os
import time
import csv
import requests
import threading
import pyttsx3
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
        with open(image_path,"rb") as photo:

            files={"photo":photo}

            data={
                "chat_id":CHAT_ID,
                "caption":caption
            }

            response=requests.post(
                url,
                data=data,
                files=files,
                timeout=10
            )

            print("[TELEGRAM SENT]")

    except Exception as e:
        print("[TELEGRAM ERROR]",e)


# =========================================
# VOICE ENGINE
# =========================================

engine = pyttsx3.init()

engine.setProperty('rate',150)
engine.setProperty('volume',1.0)

voices=engine.getProperty('voices')

engine.setProperty('voice',voices[0].id)

last_voice_time={}
VOICE_DELAY=10


def speak_alert(identity,role):

    now=time.time()

    if identity in last_voice_time:
        if now-last_voice_time[identity] < VOICE_DELAY:
            return

    if identity=="Unknown":

        message="Warning. Unauthorized person detected without helmet"

    else:

        if role!="Unknown":

            message=f"{role} {identity}, please wear your helmet"

        else:

            message=f"{identity}, please wear your helmet"

    def speak():
        try:
            engine.say(message)
            engine.runAndWait()

        except Exception as e:
            print("Voice Error:",e)

    threading.Thread(target=speak).start()

    last_voice_time[identity]=now


# =========================================
# ROLE MAPPING
# =========================================

def get_role_from_helmet(color):

    if color=="WHITE":
        return "Engineer"

    elif color=="YELLOW":
        return "Worker"

    return "Unknown"


# =========================================
# LOAD YOLO
# =========================================

model=YOLO(
r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt"
)


# =========================================
# LOAD INSIGHTFACE
# =========================================

app=FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)

app.prepare(
ctx_id=0,
det_size=(640,640)
)


# =========================================
# LOAD FACE DATABASE
# =========================================

face_db=np.load(
r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy",
allow_pickle=True
).item()

for name in face_db:
    face_db[name]=face_db[name]/np.linalg.norm(face_db[name])


# =========================================
# FACE RECOGNITION
# =========================================

def recognize_face(embedding):

    embedding=embedding/np.linalg.norm(embedding)

    best_name="Unknown"
    best_dist=999

    for name,db_emb in face_db.items():

        dist=np.linalg.norm(
        embedding-db_emb
        )

        if dist<best_dist:

            best_dist=dist
            best_name=name

    return best_name if best_dist<1.2 else "Unknown"


# =========================================
# HELMET COLOR DETECTION
# =========================================

def detect_helmet_color(crop):

    hsv=cv2.cvtColor(
    crop,
    cv2.COLOR_BGR2HSV
    )

    yellow_mask=cv2.inRange(
    hsv,
    np.array([20,100,100]),
    np.array([35,255,255])
    )

    white_mask=cv2.inRange(
    hsv,
    np.array([0,0,180]),
    np.array([180,50,255])
    )

    yellow=cv2.countNonZero(
    yellow_mask
    )

    white=cv2.countNonZero(
    white_mask
    )

    if yellow>white and yellow>500:
        return "YELLOW"

    elif white>yellow and white>500:
        return "WHITE"

    return "UNKNOWN"


# =========================================
# LOGGING
# =========================================

log_dir="violations"

os.makedirs(
log_dir,
exist_ok=True
)

csv_file=os.path.join(
log_dir,
"logs.csv"
)

if not os.path.exists(csv_file):

    with open(csv_file,"w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow(
        ["Time",
        "Name",
        "Violation",
        "Role",
        "Image"]
        )


last_alert_time={}
ALERT_DELAY=10


# =========================================
# SAVE VIOLATION
# =========================================

def save_violation(
frame,
identity,
helmet_color
):

    global last_alert_time

    now=time.time()

    if identity in last_alert_time:

        if now-last_alert_time[identity]<ALERT_DELAY:
            return

    timestamp=datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
    )

    filename=f"{identity}_{timestamp.replace(':','-')}.jpg"

    path=os.path.join(
    log_dir,
    filename
    )

    cv2.imwrite(
    path,
    frame
    )

    role=get_role_from_helmet(
    helmet_color
    )

    with open(csv_file,"a",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([
        timestamp,
        identity,
        "NO HELMET",
        role,
        path
        ])

    speak_alert(
    identity,
    role
    )

    send_telegram_alert(
    identity,
    "NO HELMET",
    timestamp,
    path
    )

    last_alert_time[identity]=now


# =========================================
# CAMERA
# =========================================

cap=cv2.VideoCapture(0)

while True:

    ret,frame=cap.read()

    if not ret:
        break

    results=model(frame)

    helmet_boxes=[]

    for r in results:

        for box in r.boxes:

            cls=int(box.cls[0])
            conf=float(box.conf[0])

            label=model.names[cls]

            x1,y1,x2,y2=map(
            int,
            box.xyxy[0]
            )

            if conf>0.5:

                crop=frame[
                y1:y2,
                x1:x2
                ]

                helmet_color=detect_helmet_color(
                crop
                )

                helmet_boxes.append({

                "box":(
                x1,
                y1,
                x2,
                y2
                ),

                "color":helmet_color

                })

    faces=app.get(frame)

    for face in faces:

        fx1,fy1,fx2,fy2=map(
        int,
        face.bbox
        )

        identity=recognize_face(
        face.embedding
        )

        cx=(fx1+fx2)//2
        cy=(fy1+fy2)//2

        has_helmet=False
        helmet_color="NONE"

        for h in helmet_boxes:

            x1,y1,x2,y2=h["box"]

            if x1-100<cx<x2+100 and y1-120<cy<y2+120:

                has_helmet=True
                helmet_color=h["color"]
                break

        if has_helmet:

            status="SAFE"
            color=(0,255,0)

        else:

            status="NO HELMET"
            color=(0,0,255)

            save_violation(
            frame,
            identity,
            helmet_color
            )

        cv2.rectangle(
        frame,
        (fx1,fy1),
        (fx2,fy2),
        color,
        2
        )

        cv2.putText(
        frame,
        f"{identity} | {status}",
        (fx1,fy1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
        )

    cv2.imshow(
    "AI Safety System",
    frame
    )

    if cv2.waitKey(1)&0xFF==27:
        break


cap.release()
cv2.destroyAllWindows()