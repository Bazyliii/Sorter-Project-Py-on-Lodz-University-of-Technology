# Imports
from multiprocessing import Process, Value as mpValue, Array as mpArray
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

# GPIO chip adress
CHIP: str = "/dev/gpiochip4"
# GPIO values multiprocessing array
gpio_values: list = mpArray("i", [0, 0, 0, 0, 0, 0, 0, 0])


def gpio_process() -> None:
    """
    GPIO process that watches GPIO lines and updates the multiprocessing array.
    """
    print("GPIO process started!")
    with gpio.request_lines(CHIP, config={}) as request:
        pass


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
        Initializes a new instance of the Hardware_PWM class.

        Args:
            channel (int): The PWM channel number.
            hz (int): The PWM frequency in Hz.
            duty_cycle (int): The PWM duty cycle in percentage.

        Returns:
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
        Changes the PWM duty cycle.

        Args:
            duty_cycle (int): The PWM duty cycle in percentage [0-100].

        Returns:
            None
        """
        self.duty_cycle = duty_cycle
        self.pwm.change_duty_cycle(self.duty_cycle)

    def change_frequency(self, hz: int) -> None:
        """
        Changes the PWM frequency.

        Args:
            hz (int): The PWM frequency in Hz.

        Returns:
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


if __name__ == "__main__":
    pass
