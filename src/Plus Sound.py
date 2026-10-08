import cv2
import numpy as np
import os
import time
import csv
import requests
import threading
import queue
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

            requests.post(
                url,
                data=data,
                files=files
            )

            print("[TELEGRAM SENT]")

    except Exception as e:

        print(e)



# =========================================
# VOICE SYSTEM
# =========================================

engine = pyttsx3.init()

engine.setProperty("rate",145)
engine.setProperty("volume",1)

voices=engine.getProperty("voices")

engine.setProperty(
    "voice",
    voices[0].id
)

speech_queue=queue.Queue()

last_voice_time={}

VOICE_INTERVAL=5


def speech_worker():

    while True:

        message=speech_queue.get()

        try:

            engine.say(message)
            engine.runAndWait()

        except Exception as e:

            print(
                "Speech Error:",
                e
            )

        speech_queue.task_done()


threading.Thread(
    target=speech_worker,
    daemon=True
).start()


def speak_alert(identity):

    now=time.time()

    if identity in last_voice_time:

        if now-last_voice_time[identity] < VOICE_INTERVAL:

            return


    if identity=="Unknown":

        message=(
            "Warning. Unauthorized person "
            "detected without helmet"
        )

    elif identity=="Cyrus":

        message=(
            "Engineer Cyrus, "
            "please wear your helmet"
        )

    else:

        message=(
            f"{identity}, "
            "please wear your helmet"
        )


    speech_queue.put(message)

    last_voice_time[identity]=now



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

    face_db[name]=(
        face_db[name]
        /
        np.linalg.norm(
            face_db[name]
        )
    )



# =========================================
# FACE RECOGNITION
# =========================================

def recognize_face(embedding):

    embedding=(
        embedding
        /
        np.linalg.norm(
            embedding
        )
    )

    best_name="Unknown"

    best_dist=999


    for name,db_emb in face_db.items():

        dist=np.linalg.norm(
            embedding-db_emb
        )

        if dist<best_dist:

            best_dist=dist

            best_name=name


    if best_dist<1.2:

        return best_name

    return "Unknown"



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


if not os.path.exists(
    csv_file
):

    with open(
        csv_file,
        "w",
        newline=""
    ) as f:

        writer=csv.writer(f)

        writer.writerow(

            [
                "Time",
                "Name",
                "Violation",
                "Image"
            ]

        )



last_alert={}

ALERT_DELAY=10



# =========================================
# SAVE VIOLATION
# =========================================

def save_violation(
    frame,
    identity
):

    now=time.time()

    if identity in last_alert:

        if now-last_alert[identity] < ALERT_DELAY:

            return


    timestamp=datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


    filename=(
        f"{identity}_{timestamp}.jpg"
    )


    path=os.path.join(
        log_dir,
        filename
    )


    cv2.imwrite(
        path,
        frame
    )


    with open(
        csv_file,
        "a",
        newline=""
    ) as f:

        writer=csv.writer(f)

        writer.writerow(

            [
                timestamp,
                identity,
                "NO HELMET",
                path
            ]

        )


    send_telegram_alert(
        identity,
        "NO HELMET",
        timestamp,
        path
    )


    last_alert[identity]=now



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

            cls=int(
                box.cls[0]
            )

            conf=float(
                box.conf[0]
            )

            label=model.names[
                cls
            ]

            x1,y1,x2,y2=map(
                int,
                box.xyxy[0]
            )


            if conf>0.5:

                if label=="helmet":

                    helmet_boxes.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )


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


        for x1,y1,x2,y2 in helmet_boxes:

            if (

                x1-100 < cx < x2+100

                and

                y1-120 < cy < y2+120

            ):

                has_helmet=True

                break


        if has_helmet:

            color=(0,255,0)

            status="SAFE"

        else:

            color=(0,0,255)

            status="NO HELMET"

            speak_alert(
                identity
            )

            save_violation(
                frame,
                identity
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
        "AI Safety System FINAL",
        frame
    )


    if cv2.waitKey(1)&0xFF==27:
        break


cap.release()

cv2.destroyAllWindows()