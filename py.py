import cv2
import numpy
from enum import Enum
from flask import Response
from flask import Flask
from flask import render_template
from threading import Thread, Lock
from waitress import serve

app = Flask(__name__)


class Colors(Enum):
    RED = 0
    GREEN = 1
    ORANGE = 2
    WHITE = 3

    def get_rgb(self):
        match self:
            case Colors.RED:
                return (0, 0, 255)
            case Colors.GREEN:
                return (0, 255, 0)
            case Colors.ORANGE:
                return (0, 165, 255)
            case Colors.WHITE:
                return (200, 200, 200)


class HSVColorMask:
    def __init__(self, lower: tuple[int, int, int], upper: tuple[int, int, int]):
        self.lower: numpy.ndarray = numpy.array(lower)
        self.upper: numpy.ndarray = numpy.array(upper)

    def get_contours(self, frame: numpy.ndarray):
        color_mask: numpy.ndarray = cv2.inRange(frame, self.lower, self.upper)
        return cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


area: tuple = (300, 500, 100, 340)
radius: int = 11

red_mask: HSVColorMask = HSVColorMask((0, 70, 70), (10, 255, 255))  # Git
green_mask: HSVColorMask = HSVColorMask((45, 85, 53), (84, 255, 255))  # Git
orange_mask: HSVColorMask = HSVColorMask((10, 50, 50), (45, 255, 255))  # Git
white_mask: HSVColorMask = HSVColorMask((0, 0, 34), (180, 48, 255))  # Git

capture: cv2.VideoCapture = cv2.VideoCapture(r"/home/bazili/Desktop/Projekt_Wizja/record.avi")
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 80)
delay: int = int(1000 / capture.get(cv2.CAP_PROP_FPS))

outputFrame = None
lock = Lock()


def cv():
    global lock, outputFrame
    queue_of_elements: list = []
    tracker_types: dict[Colors, list] = {color: [] for color in Colors}

    def detector(contours, color: Colors):
        current_contours: list = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                moments: dict = cv2.moments(contour)
                if moments["m00"] != 0:
                    center_x: int = int(moments["m10"] / moments["m00"])
                    center_y: int = int(moments["m01"] / moments["m00"])
                    if (
                        center_x > area[0]
                        and center_x < area[1]
                        and center_y > area[2]
                        and center_y < area[3]
                    ):
                        cv2.circle(
                            frame, (center_x, center_y), radius, color.get_rgb(), -1
                        )
                        current_contours.append((center_x, center_y))
        for center_x, center_y in current_contours:
            is_new: bool = True
            for tracker_x, tracker_y in tracker_types[color]:
                if (
                    numpy.linalg.norm(
                        numpy.array([center_x, center_y])
                        - numpy.array([tracker_x, tracker_y])
                    )
                    < 25
                ):
                    is_new: bool = False
                    break
            if is_new:
                queue_of_elements.append(color)
        tracker_types[color] = current_contours
        

    while True:
        frame: numpy.ndarray = capture.read()[1]
        hsv_frame: numpy.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detector(red_mask.get_contours(hsv_frame), Colors.RED)
        detector(green_mask.get_contours(hsv_frame), Colors.GREEN)
        detector(orange_mask.get_contours(hsv_frame), Colors.ORANGE)
        # detector(white_mask.get_contours(hsv_frame), Colors.WHITE)


        cv2.rectangle(frame, (area[0], area[2]), (area[1], area[3]), (0, 0, 0), 3)
        # cv2.imshow("Frame", frame)

        with lock:
            outputFrame = frame.copy()
        print(queue_of_elements)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break


def generate():
    global lock, outputFrame
    while True:
        with lock:
            if outputFrame is None:
                continue
            flag, encodedImage = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    t1 = Thread(target=cv)
    t1.daemon = True
    t1.start()
    serve(app, host="0.0.0.0", port=5500)
