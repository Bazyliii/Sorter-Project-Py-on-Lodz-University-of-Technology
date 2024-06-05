import cv2
import numpy as np


def main():
    capture = cv2.VideoCapture(0, cv2.CAP_ANY)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    capture.set(cv2.CAP_PROP_FPS, 30)
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    # delay = 1
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([80, 255, 255])

    lower_orange = np.array([15, 50, 50])
    upper_orange = np.array([25, 255, 255])

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([70, 40, 255])

    lower_black = np.array([0,0,0])
    upper_black = np.array([180, 255, 50])


    reds = 0
    oranges = 0
    greens = 0
    whites = 0
    blacks = 0
    
    area = [300, 500, 100, 340]
    radius = 11

    tracker_types = {
        'red': [],
        'orange': [],
        'green': [],
        'white': [],
        'black': [],
    }

    def detector(contours, color, tracker_key):
        nonlocal reds, oranges, greens, whites, blacks
        current_contours = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                m = cv2.moments(contour)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    if cx > area[0] and cx < area[1] and cy > area[2] and cy < area[3]:
                        cv2.circle(frame, (cx, cy), radius, color, -1)
                        current_contours.append((cx, cy))

        new_contours = []
        for (cx, cy) in current_contours:
            is_new = True
            for (tx, ty) in tracker_types[tracker_key]:
                if np.linalg.norm(np.array([cx, cy]) - np.array([tx, ty])) < 20:
                    is_new = False
                    break
            if is_new:
                new_contours.append((cx, cy))
                if tracker_key == 'red':
                    reds += 1
                elif tracker_key == 'orange':
                    oranges += 1
                elif tracker_key == 'green':
                    greens += 1
                elif tracker_key == 'white':
                    whites += 1
                elif tracker_key == 'black':
                    blacks += 1

        tracker_types[tracker_key] = current_contours

    while True:
        _, frame = capture.read()
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        orange, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        white, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        black, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detector(red, (0, 0, 255), 'red')
        detector(orange, (0, 125, 225), 'orange')
        detector(green, (0, 255, 0), 'green')
        detector(white, (205, 205, 205), 'white')
        detector(black, (0, 0, 0), 'black')

        print(f"Reds: {reds}, Oranges: {oranges}, Greens: {greens}, Whites: {whites}", "Blacks: ", blacks)
        cv2.rectangle(frame, (area[0], area[2]), (area[1], area[3]), (0, 0, 0), 3)
        cv2.imshow("frame", frame)
        cv2.imshow("black", black_mask)
        cv2.imshow("hsv", hsv)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
