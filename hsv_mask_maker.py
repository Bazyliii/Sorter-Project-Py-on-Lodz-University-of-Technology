import cv2
import numpy


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 80)
delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))

def none(x):
    pass


cv2.namedWindow("Up")
cv2.namedWindow("Down")
cv2.createTrackbar("Hue", "Up", 0, 180, none)
cv2.createTrackbar("Saturation", "Up", 0, 255, none)
cv2.createTrackbar("Value", "Up", 0, 255, none)
cv2.createTrackbar("Hue", "Down", 0, 180, none)
cv2.createTrackbar("Saturation", "Down", 0, 255, none)
cv2.createTrackbar("Value", "Down", 0, 255, none)


while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, (cv2.getTrackbarPos("Hue", "Down"), cv2.getTrackbarPos("Saturation", "Down"), cv2.getTrackbarPos("Value", "Down")), (cv2.getTrackbarPos("Hue", "Up"), cv2.getTrackbarPos("Saturation", "Up"), cv2.getTrackbarPos("Value", "Up")))
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break