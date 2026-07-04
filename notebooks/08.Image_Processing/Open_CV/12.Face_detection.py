# Face detection using haarcascades
import cv2 as cv


img = cv.imread("image\persons.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier("opencv\haar_face.xml")
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7)

print(f"no of faces found = {len(faces_rect)}")

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)


cv.imshow("detected face",img)
cv.waitKey(0)
cv.destroyAllWindows()

# Limitations of Haar Cascades
# Detects non-face objects sometimes (false positives)
# Sensitive to lighting and background
# Not robust for modern real-world images