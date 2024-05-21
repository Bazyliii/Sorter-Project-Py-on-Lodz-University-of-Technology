# Imports
from enum import Enum
from multiprocessing import Process, Value as mpValue
from multiprocessing.sharedctypes import Synchronized
from rpi_hardware_pwm import HardwarePWM
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


class Button:
    def __init__(
        self,
        pin: int,
        toggle: bool = False,
    ) -> None:
        self.pin: int = pin
        self.identifier: Identifier = Identifier.Button
        self.value: Synchronized = mpValue("i", 1)
        self.toggle: bool = toggle
        self.last_state: int = 1
        
    def return_settings(self) -> tuple:
        return (
            gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
                debounce_period=timedelta(milliseconds=10),
            ),
        )

    def return_identifier(self) -> Identifier:
        return self.identifier

    def return_pin(self) -> tuple:
        return (self.pin,)

    def set_value(self, value: int) -> None:
        if self.toggle:
            if self.last_state != value:
                self.last_state = value
                if value == 0 and self.value.value  == 0:
                    self.value.value = 1
                elif value == 0 and self.value.value  == 1:
                    self.value.value = 0
        else:
            self.value.value = value

    def get_value(self) -> Synchronized:
        return self.value.value


class Encoder:
    def __init__(
        self,
        pin_A: int,
        pin_B: int,
    ) -> None:
        self.pin_A: int = pin_A
        self.pin_B: int = pin_B
        self.identifier = Identifier.Encoder
        self.value: Synchronized = mpValue("i", 0)
        self.prev_pin_A_state: gpioValue = gpioValue.ACTIVE

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
        return (self.pin_A, self.pin_B)

    def return_identifier(self) -> Identifier:
        return self.identifier

    def change_value(self, value: int) -> None:
        self.value.value += value

    def get_value(self) -> int:
        return self.value.value


button_1: Button = Button(2, True)
button_2: Button = Button(17, False)
encoder_1: Encoder = Encoder(3, 4)

# Dictionary of GPIO elements
GPIO_ELEMENTS: dict = {Identifier.Button: {button_1, button_2}, Identifier.Encoder: {encoder_1}}


# GPIO chip adress
CHIP: str = "/dev/gpiochip4"


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
            for settings in set.union(*GPIO_ELEMENTS.values())
            for pin in settings.return_pin()
            for setting in settings.return_settings()
        },
    ) as request:
        while True:
            for button in GPIO_ELEMENTS[Identifier.Button]:
                button.set_value(
                    gpio_value_to_numeric(request.get_value(button.return_pin()[0]))
                )

            for encoder in GPIO_ELEMENTS[Identifier.Encoder]:
                pin_A_state = request.get_value(encoder.return_pin()[0])
                if (
                    pin_A_state != encoder.prev_pin_A_state
                    and pin_A_state == gpioValue.ACTIVE
                ):
                    if request.get_value(encoder.return_pin()[1]) == gpioValue.ACTIVE:
                        encoder.change_value(-1)
                    else:
                        encoder.change_value(1)
                encoder.prev_pin_A_state = pin_A_state
            sleep(0.001)


def openCV_process() -> None:
    print("OpenCV process started!")
    pass


def robot_control_process() -> None:
    print("Robot control process started!")
    pass


class Hardware_PWM:
    def __init__(self, channel: int, hz: int, duty_cycle: int) -> None:
        self.channel: int = channel
        self.hz: int = hz
        self.duty_cycle: int = duty_cycle
        self.pwm: HardwarePWM = HardwarePWM(
            pwm_channel=self.channel, hz=self.hz, chip=2
        )

    def start(self) -> None:
        self.pwm.start(self.duty_cycle)

    def stop(self) -> None:
        self.pwm.stop()

    def change_duty_cycle(self, duty_cycle: int) -> None:
        self.duty_cycle = duty_cycle
        self.pwm.change_duty_cycle(self.duty_cycle)

    def change_frequency(self, hz: int) -> None:
        self.hz = hz
        self.pwm.change_frequency(self.hz)


class Software_PWM:
    def __init__(self, pin: int, hz: int, duty_cycle: int) -> None:
        self.pin: int = pin
        self.hz: int = hz
        self.duty_cycle: int = duty_cycle

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def change_duty_cycle(self, duty_cycle: int) -> None:
        pass

    def change_frequency(self, hz: int) -> None:
        pass


def print_process() -> None:
    print("Print process started!")
    while True:
        print(
            f"BUTTON_1: {button_1.get_value()} | " f"ENCODER_1: {encoder_1.get_value()} | " f"BUTTON_2: {button_2.get_value()}"
        )
        sleep(0.5)


if __name__ == "__main__":
    p1 = Process(target=gpio_process)
    p2 = Process(target=print_process)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
