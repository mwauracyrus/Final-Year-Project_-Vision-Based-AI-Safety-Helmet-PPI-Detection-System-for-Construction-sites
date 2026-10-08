import cv2
import numpy as np
from insightface.app import FaceAnalysis

# =========================================
# CONFIGURATION
# =========================================
# Paths matching your directory structure
FACE_DB_PATH = r"F:/Documents/yolov5safetyhelmet-main (1)/face_db.npy"
BUFFALO_ROOT = r"F:/Documents/yolov5safetyhelmet-main (1)/buffalo"

# =========================================
# INITIALIZE INSIGHTFACE MODEL
# =========================================
face_app = FaceAnalysis(name="buffalo_l", root=BUFFALO_ROOT)
# Target setting 320x320 keeps tracking rapid and lightweight
face_app.prepare(ctx_id=0, det_size=(320, 320))

# =========================================
# LOAD AND NORMALIZE FACE DATABASE
# =========================================
print("Loading face database reference dictionary...")
face_db = np.load(FACE_DB_PATH, allow_pickle=True).item()

# Pre-normalize embeddings in your dictionary for faster distance matching calculations
for key in face_db:
    face_db[key] = face_db[key] / np.linalg.norm(face_db[key])

# =========================================
# VECTOR MATH MATCHING FUNCTION
# =========================================
def recognize_identity(embedding):
    # Normalize the current runtime frame embedding vector
    embedding = embedding / np.linalg.norm(embedding)
    best_match_name = "Unknown"
    lowest_distance = 999

    # Calculate L2 distance against known profiles
    for name, db_embedding in face_db.items():
        distance = np.linalg.norm(embedding - db_embedding)
        if distance < lowest_distance:
            lowest_distance = distance
            best_match_name = name

    # Absolute distance cutoff threshold rule (1.20)
    if lowest_distance < 1.2:
        return f"{best_match_name} ({lowest_distance:.2f})"
    else:
        return f"Unknown (Dist: {lowest_distance:.2f})"

# =========================================
# LIVE STREAM TEST LOOP
# =========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Keep tracking real-time frame drops disabled

print("\nFace Analysis Stream Activated... Press 'ESC' on the UI Window to Quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to pull frame image data stream.")
        break

    # Extract all facial structures detected in the raw frame layout
    detected_faces = face_app.get(frame)

    for face in detected_faces:
        # Pull box edge positions
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Cross reference identity matching math metrics
        identity_label = recognize_identity(face.embedding)

        # Draw a bounding bounding box around detected face instances
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Overlay identity label text above boundary tracking lines
        cv2.putText(frame, identity_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Output window renderer display call
    cv2.imshow("STANDALONE INSIGHTFACE ANALYSIS RUNNER", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()