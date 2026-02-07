import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

# -----------------------------
# Load CNN model
# -----------------------------
model = tf.keras.models.load_model("gesture9_model.h5")

classes = [
    "gethome/call",
    "left/fist",
    "stop/five",
    "back/four",
    "showdance/nice",
    "up/one",
    "right/rock",
    "front/three",
    "down/two"
]

# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Parameters
# -----------------------------
BUFFER_SIZE = 10
STABLE_FRAMES = 5
ENTER_CONF = 0.90
EXIT_CONF = 0.35
PROGRAM_TIMEOUT = 30

gesture_buffer = deque(maxlen=BUFFER_SIZE)

# -----------------------------
# State variables
# -----------------------------
mode = "NORMAL"
last_activity_time = time.time()
last_executed = None
latched_command = "IDLE"

cap = cv2.VideoCapture(0)

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    gesture = "NO HAND"
    confidence = 0.0
    stable = False

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        xs = [int(lm.x * w) for lm in hand]
        ys = [int(lm.y * h) for lm in hand]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]

        if roi.size != 0:
            roi = cv2.resize(roi, (128,128))
            img = roi / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)[0]
            class_id = np.argmax(pred)
            confidence = float(pred[class_id])
            gesture = classes[class_id]

            gesture_buffer.append(gesture)
            stable = gesture_buffer.count(gesture) >= STABLE_FRAMES

            now = time.time()

            # ---------------- NORMAL MODE ----------------
            if mode == "NORMAL":
                latched_command = "IDLE"

                if gesture == "showdance/nice" and stable and confidence >= ENTER_CONF:
                    mode = "PROGRAM"
                    last_activity_time = now
                    last_executed = None
                    latched_command = "PROGRAM ACTIVE"

            # ---------------- PROGRAM MODE ----------------
            elif mode == "PROGRAM":

                # Auto timeout
                if now - last_activity_time > PROGRAM_TIMEOUT:
                    mode = "NORMAL"
                    latched_command = "AUTO EXIT (TIMEOUT)"

                elif gesture == "left/fist" and stable and confidence >= EXIT_CONF:
                    mode = "NORMAL"
                    latched_command = "EXIT PROGRAM MODE"

                elif stable:
                    last_activity_time = now

                    if gesture == "up/one" and last_executed != "ROTATE":
                        latched_command = "ROTATE IN POSITION"
                        last_executed = "ROTATE"

                    elif gesture == "down/two" and last_executed != "STABLE":
                        latched_command = "STABILIZE / HOVER"
                        last_executed = "STABLE"

                    elif gesture == "front/three" and last_executed != "LAND":
                        latched_command = "LAND ON X"
                        last_executed = "LAND"

        # Draw FOLLOWING rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    # -----------------------------
    # UI Overlay
    # -----------------------------
    cv2.putText(frame, f"MODE: {mode}",
                (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(frame, f"Gesture: {gesture}",
                (30,75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                (30,105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"Stable: {stable}",
                (30,135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.putText(frame, f"Command: {latched_command}",
                (30,170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    if mode == "PROGRAM":
        remaining = int(PROGRAM_TIMEOUT - (time.time() - last_activity_time))
        cv2.putText(frame, f"Program Timeout: {remaining}s",
                    (30,205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Gesture Program Control (LATCHED)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
