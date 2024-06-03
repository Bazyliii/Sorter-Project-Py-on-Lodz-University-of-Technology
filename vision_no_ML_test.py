import cv2
import numpy as np

last_sign = 0

def main():
    capture = cv2.VideoCapture(r"record.avi")
    capture.set(cv2.CAP_PROP_FPS, 30)
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([80, 255, 255])
    lower_orange = np.array([15, 50, 50])
    upper_orange = np.array([25, 255, 255])
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    reds = 0
    oranges = 0
    greens = 0

    area = [300, 370, 100, 340]
    radius = 11
    def detector(contours, color) -> int:
        global last_sign
        for contour in contours:

            if cv2.contourArea(contour) > 500:
                m = cv2.moments(contour)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    cv2.circle(frame, (cx, cy), radius, color, -1)
                    if cy < area[3] and cy > area[2]:
                        if cx < area[1] and cx > area[0]:
                            distance = area[0] - cx
                            if last_sign != np.sign(distance):
                                return 1
                            last_sign = np.sign(distance)
        return 0

    while True:
        _, frame = capture.read()
        # frame = frame[0:490, 0:300]
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        orange, _ = cv2.findContours(
            orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        reds += detector(red, (0, 0, 255))
        oranges += detector(orange, (0, 165, 255))
        greens += detector(green, (0, 255, 0))

        print(f"Reds: {reds}, Oranges: {oranges}, Greens: {greens}")
        # cv2.rectangle(frame, (area[0], area[2]), (area[1], area[3]), (0, 0, 0), 2)
        cv2.line(frame, (area[0], area[2]), (area[0], area[3]), (0, 0, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
