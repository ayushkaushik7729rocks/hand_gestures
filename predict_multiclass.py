import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("gesture9_model.h5")

classes = ["gethome/call","left/fist","stop/five","back/four","showdance/nice","up/one","right/rock","front/three","down/two"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    roi = frame[100:400,100:400]
    roi = cv2.resize(roi,(128,128))
    img = roi/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    gesture = classes[class_id]

    cv2.rectangle(frame,(100,100),(400,400),(255,0,0),2)
    cv2.putText(frame,gesture,(30,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Gesture CNN", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()   