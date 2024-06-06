import cv2
import numpy


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 80)
delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))

def none(x):
    pass


cv2.namedWindow("Hue", flags=cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("HueLower", "Hue", 0, 180, none)
cv2.createTrackbar("SatLower", "Hue", 0, 255, none)
cv2.createTrackbar("ValueLower", "Hue", 0, 255, none)
cv2.createTrackbar("HueUpper", "Hue", 180, 180, none)
cv2.createTrackbar("SatUpper", "Hue", 255, 255, none)
cv2.createTrackbar("ValueUpper", "Hue", 255, 255, none)


while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = numpy.array((cv2.getTrackbarPos("HueLower", "Hue"), cv2.getTrackbarPos("SatLower", "Hue"), cv2.getTrackbarPos("ValueLower", "Hue")))
    upper = numpy.array((cv2.getTrackbarPos("HueUpper", "Hue"), cv2.getTrackbarPos("SatUpper", "Hue"), cv2.getTrackbarPos("ValueUpper", "Hue")))
    mask = cv2.inRange(hsv_frame, lower, upper)
    cv2.imshow("Frame", frame)
    cv2.imshow("Hue", mask)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break