import cv2
import numpy as np


face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detector(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye.detectMultiScale(roi_gray,1.2,2)

        for ex,ey,ew,eh in eyes:
            cv2.circle(roi_color,(ex+30,ey+30),20,(0,0,255),2)
        
    return img

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    # now we are going to flip the frames
    frame = cv2.flip(frame,2)
    
    frame = cv2.resize(frame,(800,600))
    
    cv2.imshow('video',detector(frame))
    
    if cv2.waitKey(1)==13: # 13 is used for the enter key and 27 is used for the escape key
        break
        
cap.release()
cv2.destroyAllWindows()
