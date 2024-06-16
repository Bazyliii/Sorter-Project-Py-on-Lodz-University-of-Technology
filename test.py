from multiprocessing import Process, Value

from sorter_rewrite import *

pin_3 = ContinousSignal(2)
pin_5 = ContinousSignal(3)
pin_7 = ContinousSignal(4)
pin_11 = ContinousSignal(17)
pin_13 = ContinousSignal(27)
pin_15 = ContinousSignal(22)
pin_19 = ContinousSignal(10)
pin_21 = ContinousSignal(9)
pin_23 = ContinousSignal(11)
pin_29 = ContinousSignal(5)
pin_31 = ContinousSignal(6)
pin_35 = ContinousSignal(19)
pin_37 = ContinousSignal(26)
pin_8 = ContinousSignal(14)
pin_10 = ContinousSignal(15)
pin_12 = ContinousSignal(18)
pin_18 = ContinousSignal(24)
pin_22 = ContinousSignal(25)
pin_24 = ContinousSignal(7)
pin_26 = ContinousSignal(8)
pin_32 = ContinousSignal(12)
pin_36 = ContinousSignal(16)
pin_38 = ContinousSignal(20)
pin_40 = ContinousSignal(21)

pin_list: dict[int, ContinousSignal] = {3: pin_3,
                                        5: pin_5,
                                        7: pin_7,
                                        11: pin_11,
                                        13: pin_13,
                                        15: pin_15,
                                        19: pin_19,
                                        21: pin_21,
                                        23: pin_23,
                                        29: pin_29,
                                        31: pin_31,
                                        35: pin_35,
                                        37: pin_37,
                                        8: pin_8,
                                        10: pin_10,
                                        12: pin_12,
                                        18: pin_18,
                                        22: pin_22,
                                        24: pin_24,
                                        26: pin_26,
                                        32: pin_32,
                                        36: pin_36,
                                        38: pin_38,
                                        40: pin_40}

xd = Value("i", 2)

def test_process() -> None:
    for signal in pin_list.values():
        signal.set_value(False)

    for pin, signal in pin_list.items():
        print(f"Testujesz pin: {pin}")
        signal.set_value(True)
        while True:
            if xd.value == 1:
                xd.value = 0
                signal.set_value(False)
                break
            sleep(0.1)


def main() -> None:
    p1 = Process(target=gpio_process)
    p2 = Process(target=test_process)

    p1.start()
    p2.start()

    while True:
        if not p2.is_alive():
            p1.terminate()
            break
        inp: str = input("Przyciskiem Enter zmieniasz pin | Wpisz 'q' aby wyjść\n")
        if not inp:
            xd.value = 1
        if inp == "q":
            p1.terminate()
            p2.terminate()
            break
        sleep(0.1)


if __name__ == "__main__":
    main()
