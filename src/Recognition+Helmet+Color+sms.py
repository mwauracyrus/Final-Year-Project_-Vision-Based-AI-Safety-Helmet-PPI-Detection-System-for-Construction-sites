import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import africastalking

# =========================================
# AFRICA'S TALKING SETUP
# =========================================
USERNAME = "sandbox"     # or your live username
API_KEY = os.environ.get("AT_API_KEY", "")

africastalking.initialize(USERNAME, API_KEY)

sms = africastalking.SMS

TARGET_NUMBER = [os.environ.get("ALERT_PHONE_NUMBER", "")]

# =========================================
# SMS ALERT FUNCTION
# =========================================
def send_sms_alert(identity, violation, timestamp):

    message = (
        f"🚨 SAFETY ALERT 🚨\n"
        f"Name: {identity}\n"
        f"Violation: {violation}\n"
        f"Time: {timestamp}"
    )

    try:

        response = sms.send(
            message,
            TARGET_NUMBER
        )

        print("\n===== SMS SENT =====")
        print(response)
        print("====================\n")

    except Exception as e:
        print("\n===== SMS ERROR =====")
        print(e)
        print("=====================\n")


# =========================================
# LOAD YOLO MODEL
# =========================================
model = YOLO(
    r"F:/Documents/yolov5safetyhelmet-main (1)/Training5/best.pt"
)

# =========================================
# LOAD INSIGHTFACE
# =========================================
app = FaceAnalysis(
    name="buffalo_l",
    root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"
)

app.prepare(ctx_id=0, det_size=(640,640))

# =========================================
# LOAD FACE DATABASE
# =========================================
face_db = np.load(
    r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy",
    allow_pickle=True
).item()

for name in face_db:
    face_db[name] /= np.linalg.norm(face_db[name])

# =========================================
# LOGGING SETUP
# =========================================
log_dir = "violations"

os.makedirs(log_dir, exist_ok=True)

csv_file = os.path.join(
    log_dir,
    "logs.csv"
)

if not os.path.exists(csv_file):

    with open(csv_file,"w",newline="") as f:

        writer=csv.writer(f)

        writer.writerow([
            "Time",
            "Name",
            "Violation",
            "Helmet_Color",
            "Image"
        ])

# =========================================
# FACE RECOGNITION
# =========================================
def recognize_face(embedding):

    embedding = embedding / np.linalg.norm(
        embedding
    )

    min_dist = float("inf")
    identity = "Unknown"

    for name, db_emb in face_db.items():

        dist=np.linalg.norm(
            embedding-db_emb
        )

        if dist<min_dist:

            min_dist=dist
            identity=name

    if min_dist>1.2:
        identity="Unknown"

    return identity


# =========================================
# HELMET COLOR DETECTION
# =========================================
def detect_helmet_color(helmet_crop):

    hsv = cv2.cvtColor(
        helmet_crop,
        cv2.COLOR_BGR2HSV
    )

    yellow_mask = cv2.inRange(
        hsv,
        np.array([20,100,100]),
        np.array([35,255,255])
    )

    white_mask = cv2.inRange(
        hsv,
        np.array([0,0,180]),
        np.array([180,50,255])
    )

    yellow_pixels = cv2.countNonZero(
        yellow_mask
    )

    white_pixels = cv2.countNonZero(
        white_mask
    )

    if yellow_pixels > white_pixels and yellow_pixels > 500:
        return "YELLOW"

    elif white_pixels > yellow_pixels and white_pixels > 500:
        return "WHITE"

    else:
        return "UNKNOWN"


# =========================================
# VIOLATION SAVE + DEBUG
# =========================================
last_save_time = 0
save_delay = 5


def save_violation(frame, identity):

    global last_save_time

    current_time = time.time()

    if current_time - last_save_time < save_delay:

        print("[SKIPPED] Cooldown active")

        return

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    filename = (
        f"{identity}_"
        f"{timestamp.replace(':','-')}.jpg"
    )

    path = os.path.join(
        log_dir,
        filename
    )

    cv2.imwrite(
        path,
        frame
    )

    print("[DEBUG] Image saved")

    with open(csv_file,"a",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([

            timestamp,
            identity,
            "NO HELMET",
            "NONE",
            path

        ])

    print("[DEBUG] CSV saved")

    try:

        print("[DEBUG] Sending SMS...")

        send_sms_alert(
            identity,
            "NO HELMET",
            timestamp
        )

        print(
            "[DEBUG] SMS function completed"
        )

    except Exception as e:

        print(
            "[DEBUG SMS ERROR]",
            e
        )

    last_save_time = current_time


# =========================================
# START CAMERA
# =========================================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    helmet_boxes=[]

    # =====================================
    # HELMET DETECTION
    # =====================================
    for r in results:

        for box in r.boxes:

            cls=int(box.cls[0])

            conf=float(box.conf[0])

            label=model.names[cls]

            x1,y1,x2,y2=map(
                int,
                box.xyxy[0]
            )

            if label=="class_0" and conf>0.5:

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

                display = f"{helmet_color} HELMET"

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    display,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2
                )

    # =====================================
    # FACE DETECTION
    # =====================================
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

        for h in helmet_boxes:

            x1,y1,x2,y2 = h["box"]

            if (
                x1-100 < cx < x2+100
                and
                y1-120 < cy < y2+120
            ):

                has_helmet=True
                break

        if has_helmet:

            status="SAFE"
            color=(0,255,0)

        else:

            status="NO HELMET"
            color=(0,0,255)

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
        "AI Safety System + SMS Debug",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()