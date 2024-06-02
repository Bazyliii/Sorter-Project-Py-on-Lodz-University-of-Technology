# Imports
from enum import Enum
from multiprocessing import Process, Value as mpValue
from multiprocessing.sharedctypes import Synchronized
from threading import Thread
from numpy import linspace
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


class Identifier(Enum):
    Button = 0
    Encoder = 1
    Sensor = 2
    Pulse = 3
    ContinuousSignal = 4


CHIP: str = "/dev/gpiochip4"
GPIO_ELEMENTS: dict[Identifier, list] = {identifier: [] for identifier in Identifier}


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
        for i in linspace(self._start_hz, self._max_hz, 100, dtype=int):
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


pwm_1: HardwarePWM = HardwarePWM(0, 100, 95, True, 1300)
pwm_2: HardwarePWM = HardwarePWM(1, 500, 95)
pulse_1: Pulse = Pulse(14)
continous_signal_1: ContinuousSignal = ContinuousSignal(20)
button_1: Button = Button(2, False, lambda: pulse_1.set_counter(1000))
button_2: Button = Button(3, False, pwm_1.stop)
button_3: Button = Button(15, False, lambda: continous_signal_1.set_value(True))
button_4: Button = Button(16, False, lambda: continous_signal_1.set_value(False))
encoder_1: Encoder = Encoder(4, 17, 20, pwm_1.change_frequency)


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
        config={
            pin: setting
            for settings_list in GPIO_ELEMENTS.values()
            for settings in settings_list
            for pin in settings.return_pin()
            for setting in settings.return_settings()
        },
    ) as request:
        while True:
            ##################################################################################
            for button in GPIO_ELEMENTS[Identifier.Button]:
                button.set_value(
                    gpio_value_to_numeric(request.get_value(button.return_pin()[0]))
                )
            ##################################################################################
            for sensor in GPIO_ELEMENTS[Identifier.Sensor]:
                sensor.set_value(
                    gpio_value_to_numeric(request.get_value(sensor.return_pin()[0]))
                )
            ##################################################################################
            for encoder in GPIO_ELEMENTS[Identifier.Encoder]:
                pin_A_state = request.get_value(encoder.return_pin()[0])
                if (
                    pin_A_state != encoder.get_prev_pin_A_state()
                    and pin_A_state == gpioValue.ACTIVE
                ):
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
                    request.set_value(
                        continous_signal.return_pin()[0], gpioValue.ACTIVE
                    )
                else:
                    request.set_value(
                        continous_signal.return_pin()[0], gpioValue.INACTIVE
                    )
            ##################################################################################
            sleep(0.001)


def openCV_process() -> None:
    print("OpenCV process started!")
    pass


def robot_control_process() -> None:
    print("Robot control process started!")
    pass


def print_process() -> None:
    print("Print process started!")
    while True:
        temp_str = ""
        for index, button in enumerate(GPIO_ELEMENTS[Identifier.Button]):
            temp_str += f"Button {index}: {button.get_value()} | "
        for index, encoder in enumerate(GPIO_ELEMENTS[Identifier.Encoder]):
            temp_str += f"Encoder {index}: {encoder.get_value()} | "
        for index, sensor in enumerate(GPIO_ELEMENTS[Identifier.Sensor]):
            temp_str += f"Sensor {index}: {sensor.get_value()} | "
        print(temp_str)
        sleep(0.5)


if __name__ == "__main__":
    try:
        p1 = Process(target=gpio_process)
        p2 = Process(target=print_process)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
    finally:
        pwm_1.stop()
        pwm_2.stop()
