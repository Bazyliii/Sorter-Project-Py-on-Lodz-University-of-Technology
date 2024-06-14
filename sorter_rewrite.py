# !/usr/bin/env python3
import multiprocessing as mp
from datetime import timedelta
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
__version__ = "1.0.0"
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
        self.__channel: int = channel
        self.__start_hz: float = start_hz
        self.__duty_cycle: float = duty_cycle
        self.__ramp: bool = ramp
        self.__max_hz: float = max_hz
        self.__pwm: HwPWM = HwPWM(pwm_channel=self.__channel, hz=self.__start_hz, chip=2)

    def start(self) -> None:
        self.__pwm.start(self.__duty_cycle)
        if self.__ramp:

            def __ramp() -> None:
                for i in linspace(self.__start_hz, self.__max_hz, 100, dtype=float):
                    self.__pwm.change_frequency(i)
                    sleep(1 / i)

            Thread(target=__ramp).start()
        else:
            self.__pwm.change_frequency(self.__max_hz)

    def stop(self) -> None:
        self.__pwm.change_frequency(self.__start_hz)
        self.__pwm.stop()

    def change_duty_cycle(self, duty_cycle: float) -> None:
        if 0 <= duty_cycle <= 100:
            self.__duty_cycle = duty_cycle
            self.__pwm.change_duty_cycle(duty_cycle)

    def change_frequency(self, hz: float) -> None:
        if 0.1 <= hz <= self.__max_hz:
            self.__pwm.change_frequency(hz)


GPIO_ELEMENTS: list[BasicGPIO] = []

sensor_1: Sensor = Sensor(pin=2)
emergency_button_1: EmergencyButton = EmergencyButton(pin=3)
# encoder_1: Encoder = Encoder(pin_a=2, pin_b=3)
button_3: Button = Button(pin=4)
pulse_1: Pulse = Pulse(pin=14, hz=1)
pwm_1: HardwarePWM = HardwarePWM(channel=0, start_hz=0.1, max_hz=10, duty_cycle=50, ramp=True)
continous_signal_2: ContinousSignal = ContinousSignal(pin=15)
continous_signal_3: ContinousSignal = ContinousSignal(pin=18)


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
    pass


def print_process() -> None:
    while True:
        text: str = ""
        for element in GPIO_ELEMENTS:
            text += f"{element}: {element.get_numeric_value()} | "
        print(text)
        sleep(0.01)


def main() -> None:
    try:
        p1: Process = Process(target=panel_process)
        p2: Process = Process(target=gpio_process)
        p3: Process = Process(target=print_process)

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()
    finally:
        pwm_1.stop()


if __name__ == "__main__":
    main()
