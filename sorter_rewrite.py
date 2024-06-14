# !/usr/bin/env python3

import multiprocessing as mp
import time
from datetime import timedelta
from multiprocessing import Process
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from threading import Lock, Thread
from time import sleep
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


# TODO: Refactor into single class
class NonBlockingDelay:
    def __init__(self) -> None:
        self._timestamp = 0
        self._delay = 0

    def timeout(self) -> bool:
        return (millis() - self._timestamp) > self._delay

    def delay_ms(self, delay: int) -> None:
        self._timestamp: int = millis()
        self._delay: int = delay


def millis() -> int:
    return int(time.time() * 1000)


def delay_ms(delay: int) -> None:
    t0: int = millis()
    while (millis() - t0) < delay:
        pass


class BasicGPIO:
    def __init__(self, pins: tuple[int, ...]) -> None:
        self.__pins: tuple[int, ...] = pins
        self.__value: Synchronized[int] = mp.Value("i", 0)
        self.__config: tuple[LineSettings, ...] = tuple()
        GPIO_ELEMENTS.append(self)

    def return_pin(self, pin: int) -> int:
        return self.__pins[pin]

    def return_config(self) -> dict[int, LineSettings]:
        return {pin: config for pin, config in zip(self.__pins, self.__config)}

    def return_gpio_value(self) -> gpio.Value:
        return gpio.Value(self.__value.value)

    def return_numeric_value(self, multiplier: int = 1) -> int:
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

    def return_previous_pin_a_state(self) -> gpio.Value:
        return self.__previous_pin_a_state

    def set_previous_pin_a_state(self, value: gpio.Value) -> None:
        self.__previous_pin_a_state = value


# TODO Simplify delay logic
class Pulse(BasicGPIO):
    def __init__(self, pin: int, delay_timeout: int) -> None:
        super().__init__((pin,))
        super().set_config(
            LineSettings(
                direction=gpio.Direction.OUTPUT,
                output_value=gpio.Value.ACTIVE,
            )
        )
        self.__state: Synchronized[bool] = mp.Value("b", False)
        self.__delay: NonBlockingDelay = NonBlockingDelay()
        self.__delay_timeout: int = delay_timeout

    def return_state(self) -> bool:
        return self.__state.value

    def set_state(self, value: bool) -> None:
        self.__state.value = value

    def return_delay(self) -> NonBlockingDelay:
        return self.__delay

    def return_delay_timeout(self) -> int:
        return self.__delay_timeout


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
                    sleep(0.1)

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

# button_1: Button = Button(pin=2)
# button_2: Button = Button(pin=3)
encoder_1: Encoder = Encoder(pin_a=2, pin_b=3)
button_3: Button = Button(pin=4)
pulse_1: Pulse = Pulse(pin=14, delay_timeout=100)
continous_signal_2: ContinousSignal = ContinousSignal(pin=15)
continous_signal_3: ContinousSignal = ContinousSignal(pin=18)


def gpio_process() -> None:
    config = dict()
    for element in GPIO_ELEMENTS:
        config.update(element.return_config())
    with request_lines("/dev/gpiochip4", config=config) as request:
        while True:
            for element in GPIO_ELEMENTS:
                match element:
                    case Button() | EmergencyButton() | Sensor():
                        element.set_value(request.get_value(element.return_pin(0)))
                    case ContinousSignal():
                        request.set_value(element.return_pin(0), element.return_gpio_value())
                    case Encoder():
                        pin_a_state: gpio.Value = request.get_value(element.return_pin(0))
                        if pin_a_state != element.return_previous_pin_a_state() and pin_a_state == gpio.Value.ACTIVE:
                            if request.get_value(element.return_pin(1)) == gpio.Value.ACTIVE:
                                element.set_value(-1)
                            else:
                                element.set_value(1)
                        else:
                            element.set_value(0)
                        element.set_previous_pin_a_state(pin_a_state)
                    case Pulse():
                        if element.return_state() and element.return_numeric_value() > 0:
                            request.set_value(element.return_pin(0), gpio.Value(int(element.return_numeric_value() % 2 == 0)))
                            if element.return_delay().timeout():
                                element.set_value(element.return_numeric_value() - 1)
                                element.return_delay().delay_ms(element.return_delay_timeout())
                        else:
                            request.set_value(element.return_pin(0), gpio.Value.INACTIVE)
            print("XDD")


def panel_process() -> None:
    pulse_1.set_value(4000)
    pulse_1.set_state(True)
    while True:
        # sleep(4)
        print(button_3.return_numeric_value())
        sleep(0.1)


def main() -> None:
    p1: Process = Process(target=panel_process)
    p2: Process = Process(target=gpio_process)

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    main()
