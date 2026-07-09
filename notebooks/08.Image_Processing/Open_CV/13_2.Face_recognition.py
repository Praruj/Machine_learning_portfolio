import numpy as np
import cv2 as cv

haar_cascade=cv.CascadeClassifier("opencv\haar_face.xml")

features=np.load("features.npy")
labels=np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")
people = ["Hero"]


# Detect face in webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv.resize(roi, (100, 100))

        label, confidence = face_recognizer.predict(roi)

        name = people[label]

        # draw box
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text = f"{name} ({confidence:.2f})"
        cv.putText(frame, text, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


