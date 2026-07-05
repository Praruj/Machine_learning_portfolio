# Face detection with OpenCvs built in recognizer
# Pipeline : image → grayscale → detect face → crop ROI → store → label

import os
import cv2 as cv
import numpy as np


haar_cascade=cv.CascadeClassifier("opencv\haar_face.xml")
people=["Hero"]
DIR=r"Faces"

features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_arr=cv.imread(img_path)
            gray=cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)

            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

            for (x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                faces_roi = cv.resize(faces_roi, (100, 100))
                features.append(faces_roi)
                labels.append(label)

create_train()
print("training ---")

# print(f"length of features ={len(features)}")
# print(f"length of labels ={len(labels)}")

features=np.array(features)
labels=np.array(labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on fearuee list and label list
face_recognizer.train(features,labels)
face_recognizer.save("face_trained.yml")
np.save("features.npy",labels)
np.save("labels.npy",labels)
