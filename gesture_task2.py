import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------
# Load Hand Landmarker
# -------------------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)

# -------------------------
# Finger State Function
# -------------------------
def finger_states(hand):
    fingers = []

    # Thumb (x axis)
    if hand[4].x > hand[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, Middle, Ring, Pinky (y axis)
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand[tip].y < hand[tip-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers   # [thumb, index, middle, ring, pinky]

# -------------------------
# Gesture Logic
# -------------------------
def get_gesture(f):
    total = f.count(1)

    if total == 0:
        return "FIST"

    if total == 5:
        return "PALM"

    if f == [0,1,0,0,0]:
        return "ONE"

    if f == [0,1,1,0,0]:
        return "TWO"

    if f == [0,1,1,1,0]:
        return "THREE"

    if f == [1,0,0,0,0]:
        return "THUMBS_UP"

    if f == [1,1,0,0,0]:
        return "THUMBS_DOWN"

    if f == [0,1,1,0,1]:
        return "VICTORY"
    if f ==[0,1,1,1,1]:
        return "FOUR"
    if f == [1,0,0,0,1]:
        return "CALL_ME"

    if f == [0,1,0,0,1]:
        return "ROCK"

    return "UNKNOWN"

# -------------------------
# Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    gesture = "NO HAND"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        fingers = finger_states(hand)
        gesture = get_gesture(fingers)

        # Draw landmarks
        for lm in hand:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv.circle(frame, (x,y), 5, (0,255,0), -1)

    cv.putText(frame, gesture, (30,60),
                cv.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv.imshow("10 Gesture Recognition", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
