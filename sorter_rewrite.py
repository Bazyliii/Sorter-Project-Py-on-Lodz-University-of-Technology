# !/usr/bin/env python3
import multiprocessing as mp
from datetime import timedelta
from enum import Enum
from multiprocessing import Process
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from threading import Lock, Thread
from time import sleep, time
from typing import Any, Callable, Generator, Literal, NoReturn, Sequence

import gpiod.line as gpio
from gpiod import LineRequest, LineSettings, request_lines
from numpy import linspace
from rpi_hardware_pwm import HardwarePWM as HwPWM

__author__ = "Jarosław Wierzbowski"
__copyright__ = "Copyright (c) 2024, Lodz University of Technology - Sorter Project"
__credits__: list[str] = ["Jarosław Wierzbowski", "Mikołaj Ziółkowski", "Krzysztof Kazuba", "Rafał Sobala", "Rafał Arciszewski"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Jarosław Wierzbowski"
__email__ = "jaroslawierzbowski2001@gmail.com"
__status__ = "Work in progress"


class NonBlockingDelay:
    def __init__(self) -> None:
        self.__start_time: float = 0
        self.__duration: float = 0

    def is_timeout(self) -> bool:
        return (time() - self.__start_time) * 1000 > self.__duration

    def set_delay(self, duration: float) -> None:
        self.__start_time: float = time()
        self.__duration: float = duration

    @staticmethod
    def delay(duration: float) -> None:
        start_time: float = time()
        while (time() - start_time) * 1000 < duration:
            pass


# TODO: Add cleanup logic
class BasicGPIO:
    def __init__(self, pins: tuple[int, ...]) -> None:
        self.__pins: tuple[int, ...] = pins
        self.__value: Synchronized[int] = mp.Value("i", 0)
        self.__config: tuple[LineSettings, ...] = tuple()
        GPIO_ELEMENTS.append(self)

    def __str__(self) -> str:
        return str(self.__class__.__name__)

    def get_pin(self, pin: int) -> int:
        return self.__pins[pin]

    def get_config(self) -> dict[int, LineSettings]:
        return {pin: config for pin, config in zip(self.__pins, self.__config)}

    def get_gpio_value(self) -> gpio.Value:
        return gpio.Value(self.__value.value)

    def get_numeric_value(self, multiplier: int = 1) -> int:
        return self.__value.value * multiplier

    def set_value(self, value: gpio.Value | int) -> None:
        match value:
            case int():
                self.__value.value = value
            case gpio.Value():
                self.__value.value = int(value == gpio.Value.ACTIVE)
            case _:
                raise ValueError

    def set_config(self, *args: LineSettings) -> None:
        self.__config = args

    @staticmethod
    def cleanup() -> None:
        config: dict[int, LineSettings] = dict()
        for element in GPIO_ELEMENTS:
            config.update(element.get_config())
        with request_lines("/dev/gpiochip4", config=config) as request:
            for element in GPIO_ELEMENTS:
                match element:
                    case Button() | EmergencyButton() | Sensor() | ContinousSignal() | Pulse():
                        request.set_value(element.get_pin(0), gpio.Value.INACTIVE)
                    case Encoder():
                        request.set_value(element.get_pin(0), gpio.Value.INACTIVE)
                        request.set_value(element.get_pin(1), gpio.Value.INACTIVE)


class Button(BasicGPIO):
    def __init__(self, pin: int) -> None:
        super().__init__((pin,))
        super().set_config(LineSettings(edge_detection=gpio.Edge.BOTH, bias=gpio.Bias.PULL_UP, debounce_period=timedelta(milliseconds=10), active_low=True))


class ContinousSignal(BasicGPIO):
    def __init__(self, pin: int) -> None:
        super().__init__((pin,))
        super().set_config(
            LineSettings(
                direction=gpio.Direction.OUTPUT,
                output_value=gpio.Value.ACTIVE,
            )
        )


class EmergencyButton(BasicGPIO):
    def __init__(self, pin: int) -> None:
        super().__init__((pin,))
        super().set_config(
            LineSettings(
                edge_detection=gpio.Edge.BOTH,
                bias=gpio.Bias.PULL_UP,
            )
        )


class Sensor(BasicGPIO):
    def __init__(self, pin: int) -> None:
        super().__init__((pin,))
        super().set_config(LineSettings(edge_detection=gpio.Edge.BOTH, bias=gpio.Bias.PULL_UP, debounce_period=timedelta(milliseconds=10), active_low=True))


class Encoder(BasicGPIO):
    def __init__(self, pin_a: int, pin_b: int) -> None:
        super().__init__((pin_a, pin_b))
        super().set_config(
            LineSettings(
                direction=gpio.Direction.OUTPUT,
                output_value=gpio.Value.ACTIVE,
            ),
            LineSettings(
                direction=gpio.Direction.OUTPUT,
                output_value=gpio.Value.ACTIVE,
            ),
        )
        self.__previous_pin_a_state: gpio.Value = gpio.Value.ACTIVE
        self.__delay: NonBlockingDelay = NonBlockingDelay()

    def get_previous_pin_a_state(self) -> gpio.Value:
        return self.__previous_pin_a_state

    def set_previous_pin_a_state(self, value: gpio.Value) -> None:
        self.__previous_pin_a_state = value

    def set_delay(self) -> None:
        self.__delay.set_delay(10)

    def is_timeout(self) -> bool:
        return self.__delay.is_timeout()


class Pulse(BasicGPIO):
    def __init__(self, pin: int, hz: float) -> None:
        super().__init__((pin,))
        super().set_config(
            LineSettings(
                direction=gpio.Direction.OUTPUT,
                output_value=gpio.Value.ACTIVE,
            )
        )
        self.__state: Synchronized[bool] = mp.Value("b", False)
        self.__delay: NonBlockingDelay = NonBlockingDelay()
        self.__hz: float = hz

    def get_state(self) -> bool:
        return self.__state.value

    def set_state(self, value: bool) -> None:
        self.__state.value = value

    def set_delay(self) -> None:
        self.__delay.set_delay(500 / self.__hz)

    def is_timeout(self) -> bool:
        return self.__delay.is_timeout()

    def set_pulse_number(self, value: int) -> None:
        super().set_value(value * 2 + 1)


class HardwarePWM:
    def __init__(
        self,
        channel: Literal[0, 1],
        start_hz: float,
        max_hz: float,
        duty_cycle: float,
        ramp: bool = False,
    ) -> None:
        self.__channel: Literal[0, 1] = channel
        self.__start_hz: float = start_hz
        self.__duty_cycle: float = duty_cycle
        self.__ramp: bool = ramp
        self.__max_hz: float = max_hz
        self.__pwm: HwPWM = HwPWM(pwm_channel=self.__channel, hz=self.__start_hz, chip=2)

    def set_state(self, state: bool) -> None:
        if state:
            self.__start()
        else:
            self.__stop()

    def __start(self) -> None:
        self.__pwm.start(self.__duty_cycle)
        if self.__ramp:
            def __ramp() -> None:
                for i in linspace(self.__start_hz, self.__max_hz, 100, dtype=float):
                    self.__pwm.change_frequency(i)
                    sleep(1 / i)

            Thread(target=__ramp).start()
        else:
            self.__pwm.change_frequency(self.__max_hz)

    def __stop(self) -> None:
        self.__pwm.change_frequency(self.__start_hz)
        self.__pwm.stop()

    def change_duty_cycle(self, duty_cycle: float) -> None:
        if 0 <= duty_cycle <= 100:
            self.__duty_cycle = duty_cycle
            self.__pwm.change_duty_cycle(duty_cycle)

    def change_frequency(self, hz: float) -> None:
        if 0.1 <= hz <= self.__max_hz:
            self.__pwm.change_frequency(hz)

    @staticmethod
    def cleanup() -> None:
        HwPWM(0, chip=2, hz=1).stop()
        HwPWM(1, chip=2, hz=1).stop()


class State(Enum):
    STOP = 0
    START = 1
    RESET = 2
    ERROR = 3


class ProgramState:
    def __init__(self, start_state: State) -> None:
        self.__state: Synchronized[int] = mp.Value("i", start_state.value)

    def __str__(self) -> str:
        return str(State(self.__state.value))

    def get_state(self) -> State:
        return State(self.__state.value)

    def set_state(self, value: State) -> None:
        self.__state.value = State(value).value


program_state: ProgramState = ProgramState(start_state=State.STOP)


GPIO_ELEMENTS: list[BasicGPIO] = []

continous_signal_1: ContinousSignal = ContinousSignal(pin=2)
pulse_1: Pulse = Pulse(pin=3, hz=1)
continous_signal_3: ContinousSignal = ContinousSignal(pin=4)
continous_signal_4: ContinousSignal = ContinousSignal(pin=14)
continous_signal_5: ContinousSignal = ContinousSignal(pin=15)


def gpio_process() -> NoReturn:
    config: dict[int, LineSettings] = dict()
    for element in GPIO_ELEMENTS:
        config.update(element.get_config())
    with request_lines("/dev/gpiochip4", config=config) as request:
        while True:
            for element in GPIO_ELEMENTS:
                match element:
                    case Button() | EmergencyButton() | Sensor():
                        element.set_value(request.get_value(element.get_pin(0)))
                    case ContinousSignal():
                        request.set_value(element.get_pin(0), element.get_gpio_value())
                    case Encoder():
                        pin_a_state: gpio.Value = request.get_value(element.get_pin(0))
                        if pin_a_state != element.get_previous_pin_a_state() and pin_a_state == gpio.Value.ACTIVE:
                            if request.get_value(element.get_pin(1)) == gpio.Value.ACTIVE:
                                element.set_value(-1)
                            else:
                                element.set_value(1)
                        else:
                            element.set_value(0)
                        if element.is_timeout():
                            element.set_previous_pin_a_state(pin_a_state)
                            element.set_delay()
                    case Pulse():
                        if element.get_state() and element.get_numeric_value() > 0:
                            request.set_value(element.get_pin(0), gpio.Value(int(element.get_numeric_value() % 2 == 0)))
                            if element.is_timeout():
                                element.set_value(element.get_numeric_value() - 1)
                                element.set_delay()
                        else:
                            request.set_value(element.get_pin(0), gpio.Value.INACTIVE)


def panel_process() -> None:
    program_state.set_state(State.START)
    pulse_1.set_pulse_number(5000)
    pulse_1.set_state(True)


def sorter_process() -> None:
    pass


def print_process() -> None:
    while True:
        text: str = ""
        for element in GPIO_ELEMENTS:
            text += f"{element}: {element.get_numeric_value()} | "
        print(text, program_state, sep="")
        sleep(0.01)


def robot_control_process() -> None:
    pass


def camera_process() -> None:
    pass


def main() -> None:
    p1: Process = Process(target=gpio_process)
    p2: Process = Process(target=panel_process)
    p3: Process = Process(target=print_process)
    p4: Process = Process(target=robot_control_process)
    p5: Process = Process(target=sorter_process)
    p6: Process = Process(target=camera_process)

    try:
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
    finally:
        HardwarePWM.cleanup()
        program_state.set_state(State.STOP)
        BasicGPIO.cleanup()


if __name__ == "__main__":
    main()
