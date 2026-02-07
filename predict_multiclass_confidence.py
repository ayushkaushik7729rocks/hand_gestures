import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("gesture9_model.h5")

classes = [
    "gethome/call",
    "left/fist",
    "stop/five",
    "back/four",
    "show rotation /nice",
    "up/one",
    "right/rock",
    "front/three",
    "down/two"
]

cap = cv2.VideoCapture(0)

# -----------------------------
# Confidence logic
# -----------------------------
BUFFER_SIZE = 10
CONF_THRESHOLD = 0.85
STABLE_FRAMES = 7

gesture_buffer = deque(maxlen=BUFFER_SIZE)
confidence_buffer = deque(maxlen=BUFFER_SIZE)

last_executed = None

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # ROI (static here; you can replace with auto-follow later)
    roi = frame[100:400, 100:400]
    roi = cv2.resize(roi, (128, 128))

    img = roi / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)[0]
    class_id = np.argmax(pred)
    confidence = float(pred[class_id])
    gesture = classes[class_id]

    # Store history
    gesture_buffer.append(gesture)
    confidence_buffer.append(confidence)

    # Safety check
    execute = False
    avg_conf = sum(confidence_buffer) / len(confidence_buffer)

    if gesture_buffer.count(gesture) >= STABLE_FRAMES and avg_conf >= CONF_THRESHOLD:
        execute = True

    # Execute once per stable gesture
    if execute and gesture != last_executed:
        print(f"EXECUTED → {gesture.upper()}  (conf={avg_conf:.2f})")
        last_executed = gesture

    # -----------------------------
    # UI DRAWING
    # -----------------------------
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

    # Gesture text
    cv2.putText(frame, f"Gesture: {gesture}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Confidence text
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Average confidence text
    cv2.putText(frame, f"Avg Conf: {avg_conf*100:.1f}%",
                (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Confidence bar
    bar_x, bar_y = 30, 130
    bar_w, bar_h = 200, 20
    fill_w = int(bar_w * avg_conf)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (255, 255, 255), 2)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h),
                  (0, 255, 0) if avg_conf >= CONF_THRESHOLD else (0, 0, 255),
                  -1)

    # Status
    status_text = "SAFE (EXECUTED)" if execute else "WAITING"
    status_color = (0, 255, 0) if execute else (0, 0, 255)

    cv2.putText(frame, status_text,
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    cv2.imshow("Confidence-Aware Gesture Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
