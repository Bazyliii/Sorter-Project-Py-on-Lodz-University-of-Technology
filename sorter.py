# Imports
import cv2
import numpy
from enum import Enum
from flask import Flask, Response, render_template
from multiprocessing import Process, Value as mpValue, Queue as mpQueue
from multiprocessing.sharedctypes import Synchronized
from threading import Thread, Lock
from waitress import serve
from rpi_hardware_pwm import HardwarePWM as HwPWM
from time import sleep
from datetime import timedelta
import gpiod as gpio
from gpiod.line import (
    Direction as gpioDirection,
    Value as gpioValue,
    Bias as gpioBias,
    Edge as gpioEdge,
)


# Classes
class Color(Enum):
    RED = 0
    GREEN = 1
    ORANGE = 2
    WHITE = 3

    def get_rgb(self):
        match self:
            case Color.RED:
                return (0, 0, 255)
            case Color.GREEN:
                return (0, 255, 0)
            case Color.ORANGE:
                return (0, 165, 255)
            case Color.WHITE:
                return (200, 200, 200)


class HSVColorMask:
    def __init__(self, lower: tuple[int, int, int], upper: tuple[int, int, int]):
        self.lower: numpy.ndarray = numpy.array(lower)
        self.upper: numpy.ndarray = numpy.array(upper)

    def get_contours(self, frame: numpy.ndarray):
        color_mask: numpy.ndarray = cv2.inRange(frame, self.lower, self.upper)
        return cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


class Identifier(Enum):
    Button = 0
    Encoder = 1
    Sensor = 2
    Pulse = 3
    ContinuousSignal = 4


class Button:
    def __init__(self, pin: int, toggle: bool = False, function=None) -> None:
        self._pin: int = pin
        self._identifier: Identifier = Identifier.Button
        self._value: Synchronized[int] = mpValue("i", 1)
        self._toggle: bool = toggle
        self._last_state: int = 1
        self._function = function
        GPIO_ELEMENTS[Identifier.Button].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
                debounce_period=timedelta(milliseconds=10),
            ),
        )

    def return_identifier(self) -> Identifier:
        return self._identifier

    def return_pin(self) -> tuple:
        return (self._pin,)

    def set_value(self, value: int) -> None:
        if self._toggle:
            if self._last_state != value:
                self._last_state = value
                if value == 0 and self._value.value == 0:
                    self._value.value = 1
                elif value == 0 and self._value.value == 1:
                    self._value.value = 0
        else:
            self._value.value = value
        if self._value.value == 0 and self._function is not None:
            self._function()

    def get_value(self) -> int:
        return self._value.value


class Encoder:
    def __init__(
        self,
        pin_A: int,
        pin_B: int,
        multiplier: int = 1,
        function=None,
    ) -> None:
        self._pin_A: int = pin_A
        self._pin_B: int = pin_B
        self._identifier = Identifier.Encoder
        self._value: Synchronized[int] = mpValue("i", 0)
        self._prev_pin_A_state: gpioValue = gpioValue.ACTIVE
        self._multiplier: int = multiplier
        self._function = function
        GPIO_ELEMENTS[Identifier.Encoder].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return_pin(self) -> tuple:
        return (self._pin_A, self._pin_B)

    def return_identifier(self) -> Identifier:
        return self._identifier

    def set_value(self, value: int) -> None:
        self._value.value = value * self._multiplier
        if self._function is not None and self._value.value != 0:
            self._function(self._value.value)

    def get_value(self) -> int:
        return self._value.value

    def get_prev_pin_A_state(self) -> gpioValue:
        return self._prev_pin_A_state

    def set_prev_pin_A_state(self, value: gpioValue) -> None:
        self._prev_pin_A_state = value


class Sensor:
    def __init__(self, pin: int, function=None) -> None:
        self._pin: int = pin
        self._identifier: Identifier = Identifier.Sensor
        self._value: Synchronized[int] = mpValue("i", 0)
        self._function = function
        GPIO_ELEMENTS[Identifier.Sensor].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
            ),
        )

    def return_pin(self) -> tuple:
        return (self._pin,)

    def return_identifier(self) -> Identifier:
        return self._identifier

    def set_value(self, value: int) -> None:
        if self._value.value != value:
            self._value.value = value
            if self._function is not None:
                self._function(self._value.value)

    def get_value(self) -> int:
        return self._value.value


class HardwarePWM:
    def __init__(
        self,
        channel: int,
        start_hz: int,
        duty_cycle: int,
        ramp: bool = False,
        max_hz: int = 0,
    ) -> None:
        self._channel: int = channel
        self._start_hz: int = start_hz
        self._internal_hz: int = start_hz
        self._duty_cycle: int = duty_cycle
        self._pwm: HwPWM = HwPWM(pwm_channel=self._channel, hz=self._start_hz, chip=2)
        self._state: int = 0
        self._ramp: bool = ramp
        self._max_hz: int = max_hz

    def start(self) -> None:
        if self._state == 0:
            self._state = 1
            self._pwm.start(self._duty_cycle)
            if self._ramp:
                Thread(target=self._ramp_thread).start()

    def _ramp_thread(self) -> None:
        for i in numpy.linspace(self._start_hz, self._max_hz, 100, dtype=int):
            if self._state == 0:
                break
            self._internal_hz = i
            self._pwm.change_frequency(self._internal_hz)
            sleep(0.1)

    def stop(self) -> None:
        if self._state == 1:
            self._state = 0
            self.change_frequency(self._start_hz)
            self._internal_hz = self._start_hz
            self._pwm.stop()

    def change_duty_cycle(self, duty_cycle: int) -> None:
        if duty_cycle >= 0 and duty_cycle <= 100:
            self._duty_cycle = duty_cycle
            self._pwm.change_duty_cycle(self._duty_cycle)

    def change_frequency(self, hz: int) -> None:
        self._internal_hz += hz
        if self._internal_hz > 0:
            self._pwm.change_frequency(self._internal_hz)
        else:
            self._pwm.change_frequency(1)


class Pulse:
    def __init__(self, pin: int) -> None:
        self._pin: int = pin
        self._identifier: Identifier = Identifier.Pulse
        self._counter: Synchronized[int] = mpValue("i", 0)
        GPIO_ELEMENTS[Identifier.Pulse].append(self)
        pass

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return_pin(self) -> tuple:
        return (self._pin,)

    def return_identifier(self) -> Identifier:
        return self._identifier

    def stop(self) -> None:
        self._counter.value = 0

    def get_counter(self) -> int:
        return self._counter.value

    def set_counter(self, value: int) -> None:
        self._counter.value = value

    def start(self) -> bool:
        if self._counter.value > 0:
            return True
        else:
            return False


class ContinuousSignal:
    def __init__(self, pin: int) -> None:
        self._pin: int = pin
        self._identifier: Identifier = Identifier.ContinuousSignal
        self._value: Synchronized[bool] = mpValue("i", False)
        GPIO_ELEMENTS[Identifier.ContinuousSignal].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return_pin(self) -> tuple:
        return (self._pin,)

    def return_identifier(self) -> Identifier:
        return self._identifier

    def switch_value(self) -> None:
        self._value.value = not self._value.value

    def set_value(self, value: bool) -> None:
        self._value.value = value

    def get_value(self) -> bool:
        return self._value.value


class ElementQueue:
    def __init__(self) -> None:
        self._queue: mpQueue = mpQueue()

    def add_element(self, value: Color) -> None:
        self._queue.put(value)

    def get_element(self) -> Color:
        return self._queue.get()


# Constants
CHIP: str = "/dev/gpiochip4"
CAPTURE: str = r"/home/bazili/Desktop/Projekt_Wizja/record.avi"
GPIO_ELEMENTS: dict[Identifier, list] = {identifier: [] for identifier in Identifier}
DETECTION_AREA: tuple[int, int, int, int] = (300, 500, 100, 340)
POINT_RADIUS: int = 11
FLASK_SERVER: Flask = Flask(__name__)
FLASK_IP_PORT: tuple[str, int] = ("0.0.0.0", 5500)

element_queue: ElementQueue = ElementQueue()
pwm_1: HardwarePWM = HardwarePWM(0, 100, 95, True, 1400)
pwm_2: HardwarePWM = HardwarePWM(1, 500, 95)
# pulse_1: Pulse = Pulse(21)
# continous_signal_1: ContinuousSignal = ContinuousSignal(20)
# button_1: Button = Button(2, False, pwm_1.start)
# button_2: Button = Button(3, False, pwm_1.stop)
# button_3: Button = Button(26, False, pulse_1.stop)
# button_4: Button = Button(19, False, lambda: pulse_1.set_counter(400))
# encoder_1: Encoder = Encoder(4, 17, 20, pwm_1.change_frequency)

encoder_1: Encoder = Encoder(2, 3)  # 2-3
encoder_2: Encoder = Encoder(4, 17)  # 4-5
encoder_3: Encoder = Encoder(27, 22)  # 6-7
encoder_4: Encoder = Encoder(10, 9)  # 8-9
encoder_5: Encoder = Encoder(11, 5)  # 10-11
button_1: Button = Button(6)  # 12
button_2: Button = Button(19)  # 13
button_3: Button = Button(26)  # 14
# blue 15
# red 16
# yellow 17


def gpio_value_to_numeric(value: gpioValue) -> int:
    match value:
        case gpioValue.ACTIVE:
            return 1
        case gpioValue.INACTIVE:
            return 0
        case _:
            raise Exception("Invalid value!")


def gpio_process() -> None:
    print("GPIO process started!")

    with gpio.request_lines(
        CHIP,
        config={pin: setting for settings_list in GPIO_ELEMENTS.values() for settings in settings_list for pin in settings.return_pin() for setting in settings.return_settings()},
    ) as request:
        while True:
            ##################################################################################
            for button in GPIO_ELEMENTS[Identifier.Button]:
                button.set_value(gpio_value_to_numeric(request.get_value(button.return_pin()[0])))
            ##################################################################################
            for sensor in GPIO_ELEMENTS[Identifier.Sensor]:
                sensor.set_value(gpio_value_to_numeric(request.get_value(sensor.return_pin()[0])))
            ##################################################################################
            for encoder in GPIO_ELEMENTS[Identifier.Encoder]:
                pin_A_state = request.get_value(encoder.return_pin()[0])
                if pin_A_state != encoder.get_prev_pin_A_state() and pin_A_state == gpioValue.ACTIVE:
                    if request.get_value(encoder.return_pin()[1]) == gpioValue.ACTIVE:
                        encoder.set_value(-1)
                    else:
                        encoder.set_value(1)
                else:
                    encoder.set_value(0)
                encoder.set_prev_pin_A_state(pin_A_state)
            ##################################################################################
            for pulse in GPIO_ELEMENTS[Identifier.Pulse]:
                if pulse.start():
                    if pulse.get_counter() % 2 == 0:
                        request.set_value(pulse.return_pin()[0], gpioValue.ACTIVE)
                    else:
                        request.set_value(pulse.return_pin()[0], gpioValue.INACTIVE)
                    pulse.set_counter(pulse.get_counter() - 1)
                else:
                    request.set_value(pulse.return_pin()[0], gpioValue.ACTIVE)
            ##################################################################################
            for continous_signal in GPIO_ELEMENTS[Identifier.ContinuousSignal]:
                if continous_signal.get_value():
                    request.set_value(continous_signal.return_pin()[0], gpioValue.ACTIVE)
                else:
                    request.set_value(continous_signal.return_pin()[0], gpioValue.INACTIVE)
            ##################################################################################
            sleep(0.001)


def camera_process() -> None:
    print("OpenCV process started!")

    red_mask: HSVColorMask = HSVColorMask((0, 70, 70), (10, 255, 255))
    green_mask: HSVColorMask = HSVColorMask((45, 85, 53), (84, 255, 255))
    orange_mask: HSVColorMask = HSVColorMask((10, 50, 50), (45, 255, 255))
    white_mask: HSVColorMask = HSVColorMask((0, 0, 34), (180, 48, 255))

    capture: cv2.VideoCapture = cv2.VideoCapture(CAPTURE)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    capture.set(cv2.CAP_PROP_EXPOSURE, 80)
    frame_delay: int = int(1000 / capture.get(cv2.CAP_PROP_FPS))

    output_frame: numpy.ndarray = numpy.array(None)
    thread_lock: Lock = Lock()
    tracker_types: dict[Color, list] = {color: [] for color in Color}

    def openCV_thread():
        nonlocal thread_lock, output_frame

        def detector(contours, color: Color):
            current_contours: list = []
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    moments: dict = cv2.moments(contour)
                    if moments["m00"] != 0:
                        center_x: int = int(moments["m10"] / moments["m00"])
                        center_y: int = int(moments["m01"] / moments["m00"])
                        if center_x > DETECTION_AREA[0] and center_x < DETECTION_AREA[1] and center_y > DETECTION_AREA[2] and center_y < DETECTION_AREA[3]:
                            cv2.circle(frame, (center_x, center_y), POINT_RADIUS, color.get_rgb(), -1)
                            current_contours.append((center_x, center_y))
            for center_x, center_y in current_contours:
                is_new: bool = True
                for tracker_x, tracker_y in tracker_types[color]:
                    if numpy.linalg.norm(numpy.array([center_x, center_y]) - numpy.array([tracker_x, tracker_y])) < 25:
                        is_new: bool = False
                        break
                    if is_new:
                        element_queue.add_element(color)
                tracker_types[color] = current_contours

        while True:
            frame: numpy.ndarray = capture.read()[1]
            hvs_frame: numpy.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            detector(red_mask.get_contours(hvs_frame), Color.RED)
            detector(green_mask.get_contours(hvs_frame), Color.GREEN)
            detector(orange_mask.get_contours(hvs_frame), Color.ORANGE)
            detector(white_mask.get_contours(hvs_frame), Color.WHITE)

            cv2.rectangle(frame, (DETECTION_AREA[0], DETECTION_AREA[2]), (DETECTION_AREA[1], DETECTION_AREA[3]), (0, 0, 0), 3)
            with thread_lock:
                output_frame = frame.copy()

            sleep(frame_delay / 1000)

    def generate_frame():
        nonlocal thread_lock, output_frame
        while True:
            with thread_lock:
                if output_frame is None:
                    continue
                flag, encodedImage = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n")

    @FLASK_SERVER.route("/")
    def index():
        return render_template("index.html")

    @FLASK_SERVER.route("/video_feed")
    def video_feed():
        return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

    t1: Thread = Thread(target=openCV_thread)
    t1.daemon = True
    t1.start()
    serve(FLASK_SERVER, host=FLASK_IP_PORT[0], port=FLASK_IP_PORT[1])


def robot_control_process() -> None:
    print("Robot control process started!")
    pass


def print_process() -> None:
    print("Print process started!")
    while True:
        temp_str: str = ""
        for index, button in enumerate(GPIO_ELEMENTS[Identifier.Button]):
            temp_str += f"Button {index}: {button.get_value()} | "
        for index, encoder in enumerate(GPIO_ELEMENTS[Identifier.Encoder]):
            temp_str += f"Encoder {index}: {encoder.get_value()} | "
        for index, sensor in enumerate(GPIO_ELEMENTS[Identifier.Sensor]):
            temp_str += f"Sensor {index}: {sensor.get_value()} | "
        print(temp_str)
        sleep(0.01)


if __name__ == "__main__":
    try:
        p1: Process = Process(target=gpio_process)
        p2: Process = Process(target=print_process)
        p3: Process = Process(target=camera_process)
        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()
    finally:
        pwm_1.stop()
        pwm_2.stop()
