# Imports
from __future__ import annotations

from datetime import timedelta
from enum import Enum
from multiprocessing import Array as mpArray
from multiprocessing import Process
from multiprocessing import Queue as mpQueue
from multiprocessing import Value as mpValue
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from telnetlib import Telnet
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Generator, Literal, NoReturn, Sequence

import cv2
import gpiod as gpio
import numpy
from flask import Flask, Response, render_template
from gpiod.line import (
    Bias as gpioBias,
)
from gpiod.line import (
    Direction as gpioDirection,
)
from gpiod.line import (
    Edge as gpioEdge,
)
from gpiod.line import (
    Value as gpioValue,
)
from rpi_hardware_pwm import HardwarePWM as HwPWM
from waitress import serve


# Classes
class Color(Enum):
    RED = 0
    GREEN = 1
    ORANGE = 2
    WHITE = 3

    def get_rgb(self) -> tuple[int, int, int]:
        """Get RGB value for color."""
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
        """Initialize HSV color masking."""
        self.__lower: numpy.ndarray = numpy.array(lower)
        self.__upper: numpy.ndarray = numpy.array(upper)

    def get_contours(self, frame: numpy.ndarray) -> Sequence[cv2.Mat | numpy.ndarray[Any, numpy.dtype[numpy.integer[Any] | numpy.floating[Any]]]]:
        """Get contours for HSV color masking."""
        color_mask: numpy.ndarray = cv2.inRange(frame, self.__lower, self.__upper)
        return cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


class Identifier(Enum):
    Button = 0
    Encoder = 1
    Sensor = 2
    Pulse = 3
    ContinuousSignal = 4
    EmergencyButton = 5


class Button:
    def __init__(self, pin: int, toggle: bool = False, function: Callable | None = None) -> None:
        self.__pin: int = pin
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

    def return_pin(self) -> tuple:
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
        function: Callable | None = None,
    ) -> None:
        self.__pin_A: int = pin_A
        self.__pin_B: int = pin_B
        self.__value: Synchronized[int] = mpValue("i", 0)
        self.__prev_pin_A_state: gpioValue = gpioValue.ACTIVE
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

    def return_pin(self) -> tuple:
        return (self.__pin_A, self.__pin_B)

    def set_value(self, value: int) -> None:
        self.__value.value = value * self.__multiplier
        if self.__function is not None and self.__value.value != 0:
            self.__function(self.__value.value)

    def get_value(self) -> int:
        return self.__value.value

    def get_prev_pin_A_state(self) -> gpioValue:
        return self.__prev_pin_A_state

    def set_prev_pin_A_state(self, value: gpioValue) -> None:
        self.__prev_pin_A_state: gpioValue = value


class Sensor:
    def __init__(self, pin: int, function: Callable | None = None) -> None:
        self.__pin: int = pin
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

    def return_pin(self) -> tuple:
        return (self.__pin,)

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
        max_hz: int = 1,
    ) -> None:
        self.__channel: Synchronized[int] = mpValue("i", channel)
        self.__start_hz: Synchronized[int] = mpValue("i", start_hz)
        self.__internal_hz: Synchronized[int] = mpValue("i", start_hz)
        self.__duty_cycle: Synchronized[int] = mpValue("i", duty_cycle)
        self.__pwm: HwPWM = HwPWM(pwm_channel=self.__channel.value, hz=self.__start_hz.value, chip=2)
        self.__state: Synchronized[int] = mpValue("b", False)
        self.__ramp: Synchronized[bool] = mpValue("b", ramp)
        self.__max_hz: Synchronized[int] = mpValue("i", max_hz)

    def start(self) -> None:
        if not self.__state.value:
            self.__state.value = True
            self.__pwm.start(self.__duty_cycle.value)
            if self.__ramp.value:
                Thread(target=self.__ramp_thread).start()

    def __ramp_thread(self) -> None:
        for i in numpy.linspace(self.__start_hz.value, self.__max_hz.value, 100, dtype=int):
            if not self.__state.value:
                break
            self.__internal_hz.value = i
            self.__pwm.change_frequency(self.__internal_hz.value)
            sleep(0.1)

    def stop(self) -> None:
        if self.__state.value:
            self.__state.value = False
            self.change_frequency(self.__start_hz.value)
            self.__internal_hz.value = self.__start_hz.value
            self.__pwm.stop()

    def change_duty_cycle(self, duty_cycle: int) -> None:
        if duty_cycle >= 0 and duty_cycle <= 100:
            self.__duty_cycle.value = duty_cycle
            self.__pwm.change_duty_cycle(self.__duty_cycle.value)

    def change_frequency(self, hz: int) -> None:
        self.__internal_hz.value += hz
        if self.__internal_hz.value > 0:
            self.__pwm.change_frequency(self.__internal_hz.value)
        else:
            self.__pwm.change_frequency(1)


class Pulse:
    def __init__(self, pin: int) -> None:
        self.__pin: int = pin
        self.__state: Synchronized[bool] = mpValue("b", False)
        self.__counter: Synchronized[int] = mpValue("i", 0)
        GPIO_ELEMENTS[Identifier.Pulse].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return_pin(self) -> tuple:
        return (self.__pin,)

    def stop(self) -> None:
        self.__state.value = False
        self.__counter.value = 0

    def get_counter(self) -> int:
        return self.__counter.value

    def set_counter(self, value: int) -> None:
        self.__counter.value = value
        if self.__counter.value <= 0:
            self.__state.value = False

    def start(self) -> None:
        if self.__counter.value > 0:
            self.__state.value = True
        else:
            self.__state.value = False

    def get_pulse_state(self) -> bool:
        return self.__state.value


class ContinuousSignal:
    def __init__(self, pin: int) -> None:
        self.__pin: int = pin
        self.__value: Synchronized[bool] = mpValue("i", False)
        GPIO_ELEMENTS[Identifier.ContinuousSignal].append(self)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ),
        )

    def return_pin(self) -> tuple:
        return (self.__pin,)

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
        self.__current_position: Synchronized[int] = mpValue("i", 0)
        self.__max_pos: Synchronized[int] = mpValue("i", 0)
        self.__positioned: Synchronized[bool] = mpValue("b", False)
        self.__containters_positions: SynchronizedArray = mpArray("i", self.__num_of_containters)
        self.__working: Synchronized[bool] = mpValue("b", False)

    def add_element(self, value: Color) -> None:
        self.__queue.put(value)

    def get_element(self) -> Color:
        return self.__queue.get()

    def position(self, steps: int) -> None:
        Thread(target=self.__position_thread, args=(steps,)).start()

    def __position_thread(self, steps: int) -> None:
        if process_state.get_state() == State.Stop:
            return
        self.__direction.set_value(False)
        self.__motor.set_counter(steps)
        sleep(0.1)
        self.__motor.start()

        while self.__zero_sensor.get_value():
            sleep(0.001)
            if process_state.get_state() == State.Stop:
                return
        self.__motor.stop()
        self.__current_position.value = 0
        self.__direction.set_value(True)
        self.__motor.set_counter(steps)
        sleep(0.1)
        self.__motor.start()
        while self.__max_sensor.get_value():
            sleep(0.001)
            if process_state.get_state() == State.Stop:
                return
        self.__current_position.value = self.__max_pos.value = steps - self.__motor.get_counter()
        self.__motor.stop()
        for index, i in enumerate(numpy.linspace(0, self.__max_pos.value, self.__num_of_containters + 2, dtype=int)[1:5]):
            self.__containters_positions[index] = i
        self.__positioned.value = True
        print(self.__containters_positions[:], self.__current_position.value, self.__positioned.value)

    def __go_to_position_thread(self, position: int) -> None:
        if self.__positioned.value:
            self.__working.value = True
            if self.__current_position.value > self.__containters_positions[position]:
                self.__direction.set_value(False)
            else:
                self.__direction.set_value(True)
            self.__motor.set_counter(abs(self.__current_position.value - self.__containters_positions[position]))
            sleep(0.1)
            self.__motor.start()
            while self.__motor.get_pulse_state():
                sleep(0.001)
                if process_state.get_state() == State.Stop:
                    return
            self.__current_position.value = self.__containters_positions[position]
            self.__working.value = False
            print(self.__current_position.value)

    def go_to_position(self, position: int) -> None:
        Thread(target=self.__go_to_position_thread, args=(position,)).start()

    def get_working(self) -> bool:
        return self.__working.value


class EmergencyButton:
    def __init__(self, pin: int) -> None:
        self.__pin: int = pin
        self.__value: Synchronized[int] = mpValue("i", 0)
        GPIO_ELEMENTS[Identifier.EmergencyButton].append(self)

    def return_pin(self) -> tuple:
        return (self.__pin,)

    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
            ),
        )

    def get_value(self) -> int:
        return self.__value.value

    def set_value(self, value: int) -> None:
        self.__value.value = value


class State(Enum):
    Stop = 0
    Start = 1
    Reset = 2
    Error = 3


class ProcessState:
    def __init__(self) -> None:
        self.__state: Synchronized[int] = mpValue("i", State.Reset.value)

    def set_state(self, state: State) -> None:
        self.__state.value = state.value

    def get_state(self) -> State:
        return State(self.__state.value)


# Constants
CHIP: str = "/dev/gpiochip4"
CAPTURE: Literal[0] | str = r"C:\Users\jaros\Downloads\recordd.avi"
GPIO_ELEMENTS: dict[Identifier, list] = {identifier: [] for identifier in Identifier}
DETECTION_AREA: tuple[int, int, int, int] = (300, 500, 100, 340)
POINT_RADIUS: int = 11
FLASK_SERVER: Flask = Flask(__name__)
FLASK_IP_PORT: tuple[str, int] = ("0.0.0.0", 5500)
ROBOT_IP: str = "192.168.1.155"
ROBOT_PORT: int = 23
ROBOT_USER: str = "as"
ROBOT_CONNECTION = Telnet()

process_state: ProcessState = ProcessState()

capture: cv2.VideoCapture = cv2.VideoCapture(CAPTURE)
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, 80)

pwm_1: HardwarePWM = HardwarePWM(channel=0, start_hz=100, duty_cycle=55, ramp=True, max_hz=1300)  # PWM silnika taśmy GPIO 12

sensor_1: Sensor = Sensor(pin=22)  # Czujnik optyczny sortownika
sensor_2: Sensor = Sensor(pin=10)  # Krańcówka sortownika
sensor_3: Sensor = Sensor(pin=9)  # Krańcówka sortownika

pulse_1: Pulse = Pulse(pin=11)  # Pulsy silnika sortownika

continous_signal_1: ContinuousSignal = ContinuousSignal(pin=5)  # Kierunek silnika sortownika
continous_signal_2: ContinuousSignal = ContinuousSignal(pin=14)  # Dioda niebieskiego przycisku
continous_signal_3: ContinuousSignal = ContinuousSignal(pin=15)  # Dioda czerwonego przycisku
continous_signal_4: ContinuousSignal = ContinuousSignal(pin=18)  # Dioda żółtego przycisku


encoder_1: Encoder = Encoder(pin_A=6, pin_B=7, multiplier=20, function=pwm_1.change_frequency)  # Enkoder #1
encoder_2: Encoder = Encoder(pin_A=19, pin_B=8)  # Enkoder #2
encoder_3: Encoder = Encoder(pin_A=26, pin_B=16)  # Enkoder #3
encoder_4: Encoder = Encoder(pin_A=24, pin_B=20)  # Enkoder #4
encoder_5: Encoder = Encoder(pin_A=25, pin_B=21)  # Enkoder #5

emergency_button_1: EmergencyButton = EmergencyButton(pin=17)  # Grzyb bezpieczeństwa

sorter: Sorter = Sorter(motor=pulse_1, direction=continous_signal_1, zero_sensor=sensor_3, max_sensor=sensor_2, num_of_containters=4)


def start_process() -> None:
    if process_state.get_state() == State.Stop:
        process_state.set_state(State.Start)
        sorter.position(steps=800)
        pwm_1.start()


def stop_process() -> None:
    process_state.set_state(State.Reset)
    pwm_1.stop()
    pulse_1.stop()


def reset_process() -> None:
    if process_state.get_state() == State.Reset:
        process_state.set_state(State.Stop)


button_1: Button = Button(pin=2, toggle=False, function=start_process)  # Start
button_2: Button = Button(pin=3, toggle=False, function=stop_process)  # Stop
button_3: Button = Button(pin=4, toggle=False, function=reset_process)  # Reset


def gpio_value_to_numeric(value: gpioValue) -> int:
    match value:
        case gpioValue.ACTIVE:
            return 1
        case gpioValue.INACTIVE:
            return 0
        case _:
            raise ValueError


def gpio_process() -> None:
    print("GPIO process started!")

    with gpio.request_lines(
        CHIP,
        config={pin: setting for settings_list in GPIO_ELEMENTS.values() for settings in settings_list for pin in settings.return_pin() for setting in settings.return_settings()},
    ) as request:
        while True:
            for emergency_button in GPIO_ELEMENTS[Identifier.EmergencyButton]:
                emergency_button.set_value(gpio_value_to_numeric(request.get_value(emergency_button.return_pin()[0])))
            ##################################################################################
            for button in GPIO_ELEMENTS[Identifier.Button]:
                button.set_value(gpio_value_to_numeric(request.get_value(button.return_pin()[0])))
            ##################################################################################
            for sensor in GPIO_ELEMENTS[Identifier.Sensor]:
                sensor.set_value(gpio_value_to_numeric(request.get_value(sensor.return_pin()[0])))
            ##################################################################################
            for encoder in GPIO_ELEMENTS[Identifier.Encoder]:
                pin_A_state: gpioValue = request.get_value(encoder.return_pin()[0])
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
                if pulse.get_pulse_state():
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
            sleep(0.0015)


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

        def detector(contours: Sequence[cv2.Mat | numpy.ndarray[Any, numpy.dtype[numpy.integer[Any] | numpy.floating[Any]]]], color: Color) -> None:
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
            if process_state.get_state() == State.Start:
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
            else:
                sleep(0.1)

    def generate_frame() -> Generator[bytes, Any, NoReturn]:
        nonlocal thread_lock, output_frame
        while True:
            with thread_lock:
                if output_frame is None:
                    continue
                flag, encoded_image = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n")

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
    while True:
        if process_state.get_state() == State.Start:
            temp_str: str = ""
            for index, button in enumerate(GPIO_ELEMENTS[Identifier.Button]):
                temp_str += f"Button {index + 1}: {button.get_value()} | "
            for index, encoder in enumerate(GPIO_ELEMENTS[Identifier.Encoder]):
                temp_str += f"Encoder {index + 1}: {encoder.get_value()} | "
            for index, sensor in enumerate(GPIO_ELEMENTS[Identifier.Sensor]):
                temp_str += f"Sensor {index + 1}: {sensor.get_value()} | "
            print(temp_str)
            sleep(0.1)
        else:
            sleep(0.1)


def sorter_process() -> None:
    print("Sorter process started!")
    while True:
        if process_state.get_state() == State.Start:
            if not sorter.get_working():
                sorter.go_to_position(sorter.get_element().value)
            while sensor_1.get_value():
                sleep(0.001)
            sleep(0.75)


def panel_process() -> None:
    while True:
        if not emergency_button_1.get_value():
            stop_process()
        if process_state.get_state() == State.Reset:
            continous_signal_2.set_value(False)
            continous_signal_4.set_value(True)
            continous_signal_3.set_value(False)
        if process_state.get_state() == State.Start:
            continous_signal_3.set_value(True)
            continous_signal_2.set_value(False)
            continous_signal_4.set_value(False)
        if process_state.get_state() == State.Stop:
            continous_signal_4.set_value(False)
            continous_signal_3.set_value(False)
            continous_signal_2.set_value(True)
        sleep(0.1)


if __name__ == "__main__":
    gpio_proc = Process(target=gpio_process)
    robot_proc = Process(target=robot_control_process)
    camera_proc = Process(target=camera_process)
    sorter_proc = Process(target=sorter_process)
    print_proc = Process(target=print_process)
    panel_proc = Process(target=panel_process)

    gpio_proc.start()
    robot_proc.start()
    # camera_proc.start()
    sorter_proc.start()
    print_proc.start()
    panel_proc.start()

    gpio_proc.join()
    robot_proc.join()
    camera_proc.join()
    sorter_proc.join()
    print_proc.join()
    panel_proc.join()
