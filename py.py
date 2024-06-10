from rpi_hardware_pwm import HardwarePWM
from time import sleep

pwm = HardwarePWM(pwm_channel=0, hz=100, chip=2)
pwm.stop()
