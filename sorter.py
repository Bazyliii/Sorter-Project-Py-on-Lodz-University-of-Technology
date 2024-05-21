# Imports
from dataclasses import dataclass
from enum import Enum
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


class Identifier(Enum):
    """
    ## Quick summary:
    Enumerator of elements that can be connected to GPIO pins.
    """

    Button = 0
    Encoder = 1
    Sensor = 2


class LineSettings:
    """
    ## Quick summary:
    A class that represents a GPIO line settings.
    """

    class Button:
        """
        ## Quick summary:
        A class that represents a button.
        """
        def __init__(
            self,
            pin: int,

        ) -> None:
            """
            ## Quick summary:
            Initializes a new instance of the Button class.
            ## Args:
            - pin (int): The pin number.
            ## Returns:
                None
            """
            self.pin: int = pin
            self.identifier = Identifier.Button

        def return_settings(self) -> tuple:
            """
            ## Quick summary:
            Returns the line settings.
            ## Args:
                None
            ## Returns:
                settings (tuple): The line settings.
            """
            return (gpio.LineSettings(
                edge_detection=gpioEdge.BOTH,
                bias=gpioBias.PULL_UP,
                debounce_period=timedelta(milliseconds=10),
            ),)

        def return_identifier(self) -> Identifier:
            """
            ## Quick summary:
            Returns the identifier.
            ## Args:
                None
            ## Returns:
                identifier (Identifier): The identifier.
            """
            return self.identifier

        def return_pin(self) -> tuple:
            """
            ## Quick summary:
            Returns the pin number.
            ## Args:
                None
            ## Returns:
                pin (int): The pin number.
            """
            return (self.pin,)
    class Encoder:
        """
        ## Quick summary:
        A class that represents an encoder.
        """
        def __init__(
            self,
            pin_A: int,
            pin_B: int,
        ) -> None:
            """
            ## Quick summary:
            Initializes a new instance of the Encoder class.
            ## Args:
            - pin_A (int): The pin number for the A line.
            - pin_B (int): The pin number for the B line.
            ## Returns:
                None
            """
            self.pin_A: int = pin_A
            self.pin_B: int = pin_B
            self.identifier = Identifier.Encoder

        def return_settings(self) -> tuple:
            """
            ## Quick summary:
            Returns the line settings.
            ## Args:
                None
            ## Returns:
                settings (tuple): The line settings.
            """
            return (gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ), gpio.LineSettings(
                direction=gpioDirection.OUTPUT,
                output_value=gpioValue.ACTIVE,
            ))
        def return_pin(self) -> tuple:
            """
            ## Quick summary:
            Returns the pin numbers.
            ## Args:
                None
            ## Returns:
                pin (tuple): The pin numbers.
            """
            return (self.pin_A, self.pin_B)
        
        def return_identifier(self) -> Identifier:
            """
            ## Quick summary:
            Returns the identifier.
            ## Args:
                None
            ## Returns:
                identifier (Identifier): The identifier.
            """
            return self.identifier
        

# Tuple of GPIO elements
GPIO_ELEMENTS: tuple = (
    LineSettings.Button(2),
    LineSettings.Encoder(3, 4),
)
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
    Converts the GPIO state to a numeric value.
    ## Args:
        value (gpioValue): The GPIO state.
    ## Returns:
        numeric_value (int): The numeric value of that state.
    """
    match value:
        case gpioValue.ACTIVE:
            return 1
        case gpioValue.INACTIVE:
            return 0
        case _:
            raise Exception("Invalid value!")


def gpio_process() -> None:
    """
    GPIO process that watches GPIO lines and updates the multiprocessing array.
    """
    print("GPIO process started!")
    prev_pin_A: gpioValue = gpioValue.ACTIVE
    with gpio.request_lines(
        CHIP,
        config={
            pin: setting
            for settings in GPIO_ELEMENTS
            for pin in settings.return_pin()
            for setting in settings.return_settings()
        },
    ) as request:
        while True:
            pass
            for index, elem in enumerate(GPIO_ELEMENTS):
                match elem.return_identifier():
                    case Identifier.Button:
                        gpio_output_values[index] = gpio_value_to_numeric(
                            request.get_value(elem.return_pin()[0])
                        )
                    case Identifier.Encoder:
                        pin_A = request.get_value(elem.return_pin()[0])
                        if pin_A != prev_pin_A and pin_A == gpioValue.ACTIVE:
                            if request.get_value(elem.return_pin()[1]) == gpioValue.ACTIVE:
                                gpio_output_values[index] -= 1
                            else:
                                gpio_output_values[index] += 1
                        prev_pin_A = pin_A
                    case Identifier.Sensor:
                        pass
                    case _:
                        raise Exception("Invalid identifier!")
            sleep(0.001)


def openCV_process() -> None:
    """
    OpenCV process that detects objects and puts them in to the queue for the sorting process.
    """
    print("OpenCV process started!")
    pass


def robot_control_process() -> None:
    """
    Process that controls the Kawasaki robot.
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
    p1.join()
    p2.join()
