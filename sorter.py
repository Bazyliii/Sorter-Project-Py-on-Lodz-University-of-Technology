# Imports
import cv2
import numpy
from telnetlib import DO, ECHO, IAC, SB, TTYPE, WILL, Telnet
from typing import Any, Sequence, Generator, NoReturn
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

    def get_rgb(self) -> tuple[int, int, int]:
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
    def __init__(self, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> None:
        self.__lower: numpy.ndarray = numpy.array(lower)
        self.__upper: numpy.ndarray = numpy.array(upper)

    def get_contours(self, frame: numpy.ndarray) -> Sequence[cv2.Mat | numpy.ndarray[Any, numpy.dtype[numpy.integer[Any] | numpy.floating[Any]]]]:
        color_mask: numpy.ndarray = cv2.inRange(frame, self.__lower, self.__upper)
        return cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


class Identifier(Enum):
    Button = 0
    Encoder = 1
    Sensor = 2
    Pulse = 3
    ContinuousSignal = 4


class Button:
    def __init__(self, pin: int, toggle: bool = False, function: Any | None = None) -> None:
        self.__pin: int = pin
        self.__identifier: Identifier = Identifier.Button
        self.__value: Synchronized[int] = mpValue("i", 1)
        self.__toggle: bool = toggle
        self.__last_state: int = 1
        self.__function: Any | None = function
        GPIO_ELEMENTS[Identifier.Button].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
                debounce_period=timedelta(milliseconds=10),
            ),
        )

    def return__identifier(self) -> Identifier:
        return self.__identifier

    def return__pin(self) -> tuple:
        return (self.__pin,)

    def set_value(self, value: int) -> None:
        if self.__toggle:
            if self.__last_state != value:
                self.__last_state = value
                if value == 0 and self.__value.value == 0:
                    self.__value.value = 1
                elif value == 0 and self.__value.value == 1:
                    self.__value.value = 0
        else:
            self.__value.value = value
        if self.__value.value == 0 and self.__function is not None:
            self.__function()

    def get_value(self) -> int:
        return self.__value.value


class Encoder:
    def __init__(
        self,
        pin_A: int,
        pin_B: int,
        multiplier: int = 1,
        function: Any | None = None,
    ) -> None:
        self.__pin_A: int = pin_A
        self.__pin_B: int = pin_B
        self.__identifier = Identifier.Encoder
        self.__value: Synchronized[int] = mpValue("i", 0)
        self.__prev__pin_A_state: gpioValue = gpioValue.ACTIVE
        self.__multiplier: int = multiplier
        self.__function: Any | None = function
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

    def return__pin(self) -> tuple:
        return (self.__pin_A, self.__pin_B)

    def return__identifier(self) -> Identifier:
        return self.__identifier

    def set_value(self, value: int) -> None:
        self.__value.value = value * self.__multiplier
        if self.__function is not None and self.__value.value != 0:
            self.__function(self.__value.value)

    def get_value(self) -> int:
        return self.__value.value

    def get_prev__pin_A_state(self) -> gpioValue:
        return self.__prev__pin_A_state

    def set_prev__pin_A_state(self, value: gpioValue) -> None:
        self.__prev__pin_A_state: gpioValue = value


class Sensor:
    def __init__(self, pin: int, function: Any | None = None) -> None:
        self.__pin: int = pin
        self.__identifier: Identifier = Identifier.Sensor
        self.__value: Synchronized[int] = mpValue("i", 0)
        self.__function: Any | None = function
        GPIO_ELEMENTS[Identifier.Sensor].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
            ),
        )

    def return__pin(self) -> tuple:
        return (self.__pin,)

    def return__identifier(self) -> Identifier:
        return self.__identifier

    def set_value(self, value: int) -> None:
        if self.__value.value != value:
            self.__value.value = value
            if self.__function is not None:
                self.__function(self.__value.value)

    def get_value(self) -> int:
        return self.__value.value


class HardwarePWM:
    def __init__(
        self,
        channel: int,
        start_hz: int,
        duty_cycle: int,
        ramp: bool = False,
        max_hz: int = 0,
    ) -> None:
        self.__channel: int = channel
        self.__start_hz: int = start_hz
        self.__internal_hz: int = start_hz
        self.__duty_cycle: int = duty_cycle
        self.__pwm: HwPWM = HwPWM(pwm_channel=self.__channel, hz=self.__start_hz, chip=2)
        self.__state: int = 0
        self.__ramp: bool = ramp
        self.__max_hz: int = max_hz

    def start(self) -> None:
        if self.__state == 0:
            self.__state: int = 1
            self.__pwm.start(self.__duty_cycle)
            if self.__ramp:
                Thread(target=self.__ramp_thread).start()

    def __ramp_thread(self) -> None:
        for i in numpy.linspace(self.__start_hz, self.__max_hz, 100, dtype=int):
            if self.__state == 0:
                break
            self.__internal_hz: int = i
            self.__pwm.change_frequency(self.__internal_hz)
            sleep(0.1)

    def stop(self) -> None:
        if self.__state == 1:
            self.__state: int = 0
            self.change_frequency(self.__start_hz)
            self.__internal_hz = self.__start_hz
            self.__pwm.stop()

    def change_duty_cycle(self, duty_cycle: int) -> None:
        if duty_cycle >= 0 and duty_cycle <= 100:
            self.__duty_cycle: int = duty_cycle
            self.__pwm.change_duty_cycle(self.__duty_cycle)

    def change_frequency(self, hz: int) -> None:
        self.__internal_hz += hz
        if self.__internal_hz > 0:
            self.__pwm.change_frequency(self.__internal_hz)
        else:
            self.__pwm.change_frequency(1)


class Pulse:
    def __init__(self, pin: int) -> None:
        self.__pin: int = pin
        self.__identifier: Identifier = Identifier.Pulse
        self.__counter: Synchronized[int] = mpValue("i", 0)
        GPIO_ELEMENTS[Identifier.Pulse].append(self)
        pass

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return__pin(self) -> tuple:
        return (self.__pin,)

    def return__identifier(self) -> Identifier:
        return self.__identifier

    def stop(self) -> None:
        self.__counter.value = 0

    def get_counter(self) -> int:
        return self.__counter.value

    def set_counter(self, value: int) -> None:
        self.__counter.value = value

    def start(self) -> bool:
        if self.__counter.value > 0:
            return True
        else:
            return False


class ContinuousSignal:
    def __init__(self, pin: int) -> None:
        self.__pin: int = pin
        self.__identifier: Identifier = Identifier.ContinuousSignal
        self.__value: Synchronized[bool] = mpValue("i", False)
        GPIO_ELEMENTS[Identifier.ContinuousSignal].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return__pin(self) -> tuple:
        return (self.__pin,)

    def return__identifier(self) -> Identifier:
        return self.__identifier

    def switch_value(self) -> None:
        self.__value.value = not self.__value.value

    def set_value(self, value: bool) -> None:
        self.__value.value = value

    def get_value(self) -> bool:
        return self.__value.value


class Sorter:
    def __init__(self, motor: Pulse, direction: ContinuousSignal, zero_sensor: Sensor, max_sensor: Sensor, num_of_containters: int) -> None:
        self.__queue: mpQueue = mpQueue()
        self.__num_of_containters: int = num_of_containters
        self.__motor: Pulse = motor
        self.__direction: ContinuousSignal = direction
        self.__zero_sensor: Sensor = zero_sensor
        self.__max_sensor: Sensor = max_sensor
        self.__current_position: int = 0
        self.__max_pos: int = 0
        self.__containters_positions: list[int] = [0] * num_of_containters

    def add_element(self, value: Color) -> None:
        self.__queue.put(value)

    def get_element(self) -> Color:
        return self.__queue.get()

    def get_zero_position(self) -> None:
        self.__direction.set_value(False)
        self.__motor.set_counter(2)
        while self.__zero_sensor.get_value():
            self.__motor.set_counter(1)
            sleep(0.01)
        else:
            self.__motor.stop()
            self._current_position = 0

    def get_max_position(self) -> None:
        self.__direction.set_value(True)
        self.__motor.set_counter(2)
        while self.__max_sensor.get_value():
            self.__motor.set_counter(1)
            self.__max_pos += 1
            sleep(0.01)
        else:
            self.__motor.stop()
            self.__current_position = self.__max_pos
            self._calculate_containters_position()
    def _calculate_containters_position(self) -> None:
        self.__containters_position = numpy.linspace(0, self.__max_pos, self.__num_of_containters, dtype=int)
        print(self.__containters_position)
        pass


# Constants
CHIP: str = "/dev/gpiochip4"
CAPTURE: str = r"/home/bazili/Desktop/Projekt_Wizja/record.avi"
GPIO_ELEMENTS: dict[Identifier, list] = {identifier: [] for identifier in Identifier}
DETECTION_AREA: tuple[int, int, int, int] = (300, 500, 100, 340)
POINT_RADIUS: int = 11
FLASK_SERVER: Flask = Flask(__name__)
FLASK_IP_PORT: tuple[str, int] = ("0.0.0.0", 5500)
ROBOT_IP: str = "192.168.1.155"
ROBOT_PORT: int = 23
ROBOT_USER: str = "as"
ROBOT_CONNECTION = Telnet()


capture: cv2.VideoCapture = cv2.VideoCapture(CAPTURE)
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 80)

pwm_1: HardwarePWM = HardwarePWM(channel=0, start_hz=100, duty_cycle=95, ramp=True, max_hz=1400)  # PWM silnika taśmy GPIO 12

button_1: Button = Button(pin=2, toggle=False)  # Start
button_2: Button = Button(pin=3, toggle=False)  # Stop
button_3: Button = Button(pin=4, toggle=False)  # Reset

sensor_1: Sensor = Sensor(pin=22)  # Czujnik optyczny sortownika
sensor_2: Sensor = Sensor(pin=10)  # Krańcówka sortownika
sensor_3: Sensor = Sensor(pin=9)  # Krańcówka sortownika

pulse_1: Pulse = Pulse(pin=11)  # Pulsy silnika sortownika

continous_signal_1: ContinuousSignal = ContinuousSignal(pin=5)  # Kierunek silnika sortownika
continous_signal_2: ContinuousSignal = ContinuousSignal(pin=14)  # Dioda niebieskiego przycisku
continous_signal_3: ContinuousSignal = ContinuousSignal(pin=15)  # Dioda czerwonego przycisku
continous_signal_4: ContinuousSignal = ContinuousSignal(pin=18)  # Dioda żółtego przycisku

encoder_1: Encoder = Encoder(pin_A=6, pin_B=7)  # Enkoder #1
encoder_2: Encoder = Encoder(pin_A=19, pin_B=8)  # Enkoder #2
encoder_3: Encoder = Encoder(pin_A=26, pin_B=16)  # Enkoder #3
encoder_4: Encoder = Encoder(pin_A=24, pin_B=20)  # Enkoder #4
encoder_5: Encoder = Encoder(pin_A=25, pin_B=21)  # Enkoder #5

sorter: Sorter = Sorter(motor=pulse_1, direction=continous_signal_1, zero_sensor=sensor_2, max_sensor=sensor_3, num_of_containters=4)


def gpio_value_to_numeric(value: gpioValue) -> int:
    match value:
        case gpioValue.ACTIVE:
            return 1
        case gpioValue.INACTIVE:
            return 0
        case _:
            raise Exception("Invalid value!")


def gpio_process() -> NoReturn:
    print("GPIO process started!")

    with gpio.request_lines(
        CHIP,
        config={pin: setting for settings_list in GPIO_ELEMENTS.values() for settings in settings_list for pin in settings.return__pin() for setting in settings.return_settings()},
    ) as request:
        while True:
            ##################################################################################
            for button in GPIO_ELEMENTS[Identifier.Button]:
                button.set_value(gpio_value_to_numeric(request.get_value(button.return__pin()[0])))
            ##################################################################################
            for sensor in GPIO_ELEMENTS[Identifier.Sensor]:
                sensor.set_value(gpio_value_to_numeric(request.get_value(sensor.return__pin()[0])))
            ##################################################################################
            for encoder in GPIO_ELEMENTS[Identifier.Encoder]:
                pin_A_state: gpioValue = request.get_value(encoder.return__pin()[0])
                if pin_A_state != encoder.get_prev__pin_A_state() and pin_A_state == gpioValue.ACTIVE:
                    if request.get_value(encoder.return__pin()[1]) == gpioValue.ACTIVE:
                        encoder.set_value(-1)
                    else:
                        encoder.set_value(1)
                else:
                    encoder.set_value(0)
                encoder.set_prev__pin_A_state(pin_A_state)
            ##################################################################################
            for pulse in GPIO_ELEMENTS[Identifier.Pulse]:
                if pulse.start():
                    if pulse.get_counter() % 2 == 0:
                        request.set_value(pulse.return__pin()[0], gpioValue.ACTIVE)
                    else:
                        request.set_value(pulse.return__pin()[0], gpioValue.INACTIVE)
                    pulse.set_counter(pulse.get_counter() - 1)
                else:
                    request.set_value(pulse.return__pin()[0], gpioValue.ACTIVE)
            ##################################################################################
            for continous_signal in GPIO_ELEMENTS[Identifier.ContinuousSignal]:
                if continous_signal.get_value():
                    request.set_value(continous_signal.return__pin()[0], gpioValue.ACTIVE)
                else:
                    request.set_value(continous_signal.return__pin()[0], gpioValue.INACTIVE)
            ##################################################################################
            sleep(0.001)


def camera_process() -> None:
    print("Camera process started!")

    red_mask: HSVColorMask = HSVColorMask((0, 70, 70), (10, 255, 255))
    green_mask: HSVColorMask = HSVColorMask((45, 85, 53), (84, 255, 255))
    orange_mask: HSVColorMask = HSVColorMask((10, 50, 50), (45, 255, 255))
    white_mask: HSVColorMask = HSVColorMask((0, 0, 34), (180, 48, 255))

    frame_delay: int = int(1000 / capture.get(cv2.CAP_PROP_FPS))

    output_frame: numpy.ndarray = numpy.array(None)
    thread_lock: Lock = Lock()
    tracker_types: dict[Color, list] = {color: [] for color in Color}

    def openCV_thread() -> NoReturn:
        nonlocal thread_lock, output_frame

        def detector(contours, color: Color) -> None:
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
                        sorter.add_element(color)
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

    def generate_frame() -> Generator[bytes, Any, NoReturn]:
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
    def index() -> str:
        return render_template("index.html")

    @FLASK_SERVER.route("/video_feed")
    def video_feed() -> Response:
        return Response(generate_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

    t1: Thread = Thread(target=openCV_thread)
    t1.daemon = True
    t1.start()
    serve(FLASK_SERVER, host=FLASK_IP_PORT[0], port=FLASK_IP_PORT[1])


def robot_control_process() -> None:
    print("Robot control process started!")


def print_process() -> None:
    print("Print process started!")
    return
    while True:
        temp_str: str = ""
        for index, button in enumerate(GPIO_ELEMENTS[Identifier.Button]):
            temp_str += f"Button {index + 1}: {button.get_value()} | "
        for index, encoder in enumerate(GPIO_ELEMENTS[Identifier.Encoder]):
            temp_str += f"Encoder {index + 1}: {encoder.get_value()} | "
        for index, sensor in enumerate(GPIO_ELEMENTS[Identifier.Sensor]):
            temp_str += f"Sensor {index + 1}: {sensor.get_value()} | "
        print(temp_str)
        sleep(0.01)


def sorter_process() -> None:
    print("Sorter process started!")
    sorter.get_zero_position()
    sleep(0.1)
    sorter.get_max_position()


PROCESS_LIST: tuple[Process, ...] = (Process(target=gpio_process), Process(target=robot_control_process), Process(target=camera_process), Process(target=print_process), Process(target=sorter_process))


if __name__ == "__main__":
    try:
        for process in PROCESS_LIST:
            process.start()
        for process in PROCESS_LIST:
            process.join()
    finally:
        pwm_1.stop()
        capture.release()
