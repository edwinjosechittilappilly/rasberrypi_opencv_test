#! /usr/bin/python

import cv2
import numpy as np

# capture frames from a video
cap = cv2.VideoCapture(0)

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cascade.xml')
cars=[0];
# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
        maxSize=(100, 100),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    print cars[0]
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

    # Display frames in a window
    cv2.imshow('video2', frames)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
