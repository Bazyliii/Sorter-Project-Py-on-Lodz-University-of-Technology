# Imports
from dataclasses import dataclass
from multiprocessing import Process, Value as mpValue, Array as mpArray
from multiprocessing.sharedctypes import SynchronizedArray
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


class LineSettings:
    """
    ## Quick summary:
    A class that represents a GPIO line settings.
    """

    Button: int = 0
    Encoder: int = 1

    def __init__(
        self,
        pin: int,
        identifier: int,
        direction: gpioDirection = gpioDirection.AS_IS,
        edge_detection: gpioEdge = gpioEdge.NONE,
        bias: gpioBias = gpioBias.AS_IS,
        output_value: gpioValue = gpioValue.INACTIVE,
        debounce_period: timedelta = timedelta(),
    ) -> None:
        """
        ## Quick summary:
        Initializes a new instance of the LineSettings class.
        ## Args:
           - pin (int): The pin number.
           - direction (gpioDirection, optional): The pin direction. Defaults to gpioDirection.AS_IS.
           - edge_detection (gpioEdge, optional): The pin edge detection. Defaults to gpioEdge.NONE.
           - bias (gpioBias, optional): The pin bias. Defaults to gpioBias.AS_IS.
           - output_value (gpioValue, optional): The pin output value. Defaults to gpioValue.INACTIVE.
           - debounce_period (timedelta, optional): The pin debounce period. Defaults to timedelta().
        ## Returns:
            None
        """
        self.pin: int = pin
        self.direction: gpioDirection = direction
        self.edge_detection: gpioEdge = edge_detection
        self.bias: gpioBias = bias
        self.output_value: gpioValue = output_value
        self.debounce_period: timedelta = debounce_period
        self.identifier: int = identifier

    def return_settings(self) -> gpio.LineSettings:
        """
        ## Quick summary:
        Returns the line settings.
        ## Args:
            None
        ## Returns:
            settings (gpio.LineSettings): The line settings.
        """
        return gpio.LineSettings(
            direction=self.direction,
            edge_detection=self.edge_detection,
            bias=self.bias,
            output_value=self.output_value,
            debounce_period=self.debounce_period,
        )

    def return_identifier(self) -> int:
        return self.identifier

    def return_pin(self) -> int:
        """
        ## Quick summary:
        Returns the pin number.
        ## Args:
            None
        ## Returns:
            pin (int): The pin number.
        """
        return self.pin


BUTTON_1: LineSettings = LineSettings(
    pin=2,
    identifier=LineSettings.Button,
    edge_detection=gpioEdge.BOTH,
    bias=gpioBias.PULL_UP,
    debounce_period=timedelta(milliseconds=10),
)
BUTTON_2: LineSettings = LineSettings(
    pin=3,
    identifier=LineSettings.Button,
    edge_detection=gpioEdge.BOTH,
    bias=gpioBias.PULL_UP,
    debounce_period=timedelta(milliseconds=10),
)

# List of GPIO elements
GPIO_ELEMENTS: list[LineSettings] = [BUTTON_1, BUTTON_2]
# GPIO chip adress
CHIP: str = "/dev/gpiochip4"
# Number of inputs and outputs
INPUTS_COUNT: int = 8
OUTPUTS_COUNT: int = len(GPIO_ELEMENTS)
# GPIO multiprocessing arrays
gpio_output_values: SynchronizedArray = mpArray("i", [1] * OUTPUTS_COUNT)
gpio_input_values: SynchronizedArray = mpArray("i", [0] * INPUTS_COUNT)


def gpio_value_to_numeric(value: gpioValue) -> int:
    """
    ## Quick summary:
    Converts the GPIO value to a numeric value.
    ## Args:
        value (gpioValue): The GPIO value.
    ## Returns:
        numeric_value (int): The numeric value.
    """
    match value:
        case gpioValue.ACTIVE:
            return 1
        case gpioValue.INACTIVE:
            return 0


def gpio_process() -> None:
    """
    GPIO process that watches GPIO lines and updates the multiprocessing array.
    """
    print("GPIO process started!")
    with gpio.request_lines(
        CHIP,
        config={
            settings.return_pin(): settings.return_settings()
            for settings in GPIO_ELEMENTS
        },
    ) as request:
        while True:
            for index, elem in enumerate(GPIO_ELEMENTS):
                match elem.return_identifier():
                    case LineSettings.Button:
                        gpio_output_values[index] = gpio_value_to_numeric(
                            request.get_value(elem.return_pin())
                        )
                    case LineSettings.Encoder:
                        pass

            sleep(0.05)


def openCV_process() -> None:
    """
    OpenCV process that detects objects and puts them in to the queue for the sorting process.
    """
    print("OpenCV process started!")
    pass


def robot_control_process() -> None:
    """
    Process that controls the Kawasaki robot..
    """
    print("Robot control process started!")
    pass


class Hardware_PWM:
    """
    Simple Hardware PWM class that uses the rpi_hardware_pwm library.
    """

    def __init__(self, channel: int, hz: int, duty_cycle: int) -> None:
        """
        ## Quick summary:
        Initializes a new instance of the Hardware_PWM class.
        ## Args:
           - channel (int): The PWM channel number [0-1].
           - hz (int): The PWM frequency in Hz.
           - duty_cycle (int): The PWM duty cycle in percentage [0-100].
        ## Returns:
            None
        """
        self.channel: int = channel
        self.hz: int = hz
        self.duty_cycle: int = duty_cycle
        self.pwm: HardwarePWM = HardwarePWM(
            pwm_channel=self.channel, hz=self.hz, chip=2
        )

    def start(self) -> None:
        """
        Starts the PWM.
        """
        self.pwm.start(self.duty_cycle)

    def stop(self) -> None:
        """
        Stops the PWM.
        """
        self.pwm.stop()

    def change_duty_cycle(self, duty_cycle: int) -> None:
        """
        ## Quick summary:
        Changes the PWM duty cycle.
        ## Args:
           - duty_cycle (int): The PWM duty cycle in percentage [0-100].
        ## Returns:
            None
        """
        self.duty_cycle = duty_cycle
        self.pwm.change_duty_cycle(self.duty_cycle)

    def change_frequency(self, hz: int) -> None:
        """
        ## Quick summary:
        Changes the PWM frequency.
        ## Args:
           - hz (int): The PWM frequency in Hz.
        ## Returns:
            None
        """
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
    """
    Print process that prints the GPIO values.
    """
    print("Print process started!")
    while True:
        print(
            f"INPUT: {gpio_input_values[0:INPUTS_COUNT]} | "
            f"OUTPUT: {gpio_output_values[0:OUTPUTS_COUNT]}"
        )
        sleep(0.5)


if __name__ == "__main__":
    p1 = Process(target=gpio_process)
    p2 = Process(target=print_process)
    p1.start()
    p2.start()
