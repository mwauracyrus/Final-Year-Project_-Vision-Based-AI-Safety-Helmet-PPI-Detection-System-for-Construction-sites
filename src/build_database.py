import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l",
                   root=r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo")
app.prepare(ctx_id=-1)

faces_path = r"F:\Documents\yolov5safetyhelmet-main (1)\Training6\faces"

face_db = {}

for person in os.listdir(faces_path):
    person_path = os.path.join(faces_path, person)

    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)

        if len(faces) > 0:
            embeddings.append(faces[0].embedding)

    if len(embeddings) > 0:
        face_db[person] = np.mean(embeddings, axis=0)
        print(f"[INFO] Loaded {person}")

# Save database
np.save("face_db.npy", face_db)

print("✅ Face database saved successfully!")