import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

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

app.prepare(ctx_id=0, det_size=(640, 640))

# =========================================
# LOAD FACE DATABASE
# =========================================
face_db = np.load(
    r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy",
    allow_pickle=True
).item()

# Normalize embeddings
for name in face_db:
    face_db[name] = face_db[name] / np.linalg.norm(face_db[name])

# =========================================
# CREATE LOGGING FOLDER
# =========================================
log_dir = "violations"
os.makedirs(log_dir, exist_ok=True)

csv_file = os.path.join(log_dir, "logs.csv")

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time",
            "Name",
            "Status",
            "Helmet_Color",
            "Image"
        ])

# =========================================
# FACE RECOGNITION
# =========================================
def recognize_face(embedding):

    embedding = embedding / np.linalg.norm(embedding)

    min_dist = float("inf")
    identity = "Unknown"

    for name, db_emb in face_db.items():

        dist = np.linalg.norm(embedding - db_emb)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 1.2:
        identity = "Unknown"

    return identity


# =========================================
# HELMET COLOR DETECTION
# =========================================
def detect_helmet_color(helmet_crop):

    hsv = cv2.cvtColor(helmet_crop, cv2.COLOR_BGR2HSV)

    # -----------------------------
    # YELLOW MASK
    # -----------------------------
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    yellow_pixels = cv2.countNonZero(yellow_mask)

    # -----------------------------
    # WHITE MASK
    # -----------------------------
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 50, 255])

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    white_pixels = cv2.countNonZero(white_mask)

    # -----------------------------
    # DETERMINE COLOR
    # -----------------------------
    if yellow_pixels > white_pixels and yellow_pixels > 500:
        return "YELLOW"

    elif white_pixels > yellow_pixels and white_pixels > 500:
        return "WHITE"

    else:
        return "UNKNOWN"


# =========================================
# SAVE VIOLATION
# =========================================
last_save_time = 0
save_delay = 5

def save_violation(frame, identity):

    global last_save_time

    current_time = time.time()

    if current_time - last_save_time < save_delay:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{identity}_{timestamp}.jpg"

    path = os.path.join(log_dir, filename)

    cv2.imwrite(path, frame)

    with open(csv_file, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            identity,
            "NO HELMET",
            "NONE",
            path
        ])

    print("[VIOLATION SAVED]", filename)

    last_save_time = current_time


# =========================================
# START CAMERA
# =========================================
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =====================================
    # YOLO HELMET DETECTION
    # =====================================
    results = model(frame)

    helmet_boxes = []

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # =================================
            # HELMET DETECTED
            # =================================
            if label == "class_0" and conf > 0.5:

                helmet_crop = frame[y1:y2, x1:x2]

                helmet_color = detect_helmet_color(
                    helmet_crop
                )

                helmet_boxes.append({
                    "box": (x1, y1, x2, y2),
                    "color": helmet_color
                })

                # COLOR DISPLAY
                if helmet_color == "WHITE":
                    display = "WHITE HELMET (ENGINEER)"
                    box_color = (255,255,255)

                elif helmet_color == "YELLOW":
                    display = "YELLOW HELMET (WORKER)"
                    box_color = (0,255,255)

                else:
                    display = "HELMET"
                    box_color = (0,255,0)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    display,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

    # =====================================
    # FACE DETECTION + RECOGNITION
    # =====================================
    faces = app.get(frame)

    for face in faces:

        fx1, fy1, fx2, fy2 = map(int, face.bbox)

        identity = recognize_face(face.embedding)

        face_center_x = (fx1 + fx2) // 2
        face_center_y = (fy1 + fy2) // 2

        has_helmet = False
        helmet_type = ""

        # =================================
        # MATCH FACE TO HELMET
        # =================================
        for helmet in helmet_boxes:

            hx1, hy1, hx2, hy2 = helmet["box"]

            # Expanded matching region
            expand_x = 100
            expand_y = 120

            if (
                hx1 - expand_x < face_center_x < hx2 + expand_x
                and
                hy1 - expand_y < face_center_y < hy2 + expand_y
            ):

                has_helmet = True
                helmet_type = helmet["color"]
                break

        # =================================
        # SAFE PERSON
        # =================================
        if has_helmet:

            if helmet_type == "WHITE":
                status = "SAFE | ENGINEER"

            elif helmet_type == "YELLOW":
                status = "SAFE | WORKER"

            else:
                status = "SAFE"

            color = (0,255,0)

        # =================================
        # VIOLATION
        # =================================
        else:

            status = "NO HELMET"

            color = (0,0,255)

            save_violation(frame, identity)

        # =================================
        # DRAW FACE BOX
        # =================================
        cv2.rectangle(
            frame,
            (fx1, fy1),
            (fx2, fy2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{identity} | {status}",
            (fx1, fy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # =====================================
    # SHOW FRAME
    # =====================================
    cv2.imshow(
        "AI Construction Safety System",
        frame
    )

    # ESC TO EXIT
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()