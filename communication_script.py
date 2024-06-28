import time
from telnetlib import DO, ECHO, IAC, SB, SE, TTYPE, WILL, Telnet

def option_negotiation_callback(socket, cmd, opt):
    IS = b"\00"
    if cmd == WILL and opt == ECHO:  # hex:ff fb 01 name:IAC WILL ECHO description:(I will echo)
        socket.sendall(IAC + DO + opt)  # hex(ff fd 01), name(IAC DO ECHO), descr(please use echo)
    elif cmd == DO and opt == TTYPE:  # hex(ff fd 18), name(IAC DO TTYPE), descr(please send environment type)
        socket.sendall(IAC + WILL + TTYPE)  # hex(ff fb 18), name(IAC WILL TTYPE), descr(Dont worry, i'll send environment type)
    elif cmd == SB:
        socket.sendall(IAC + SB + TTYPE + IS + b"VT100" + IS + IAC + SE)  # hex(ff fa 18 00 b"VT100" 00 ff f0) name(IAC SB TTYPE iS VT100 iS IAC SE) descr(Start subnegotiation, environment type is VT100, end negotation)
    elif cmd == SE:  # server letting us know sub negotiation has ended
        pass  # do nothing
    else:
        print("Nieoczekiwana negocjacja telnet")

ip_address = "192.168.1.155"
port = 23
username = "as"
timeout = 5

telnet = Telnet()
telnet.set_option_negotiation_callback(option_negotiation_callback)
try:
    telnet.open(ip_address, port, timeout)
    time.sleep(0.5)
    _ = telnet.read_until(b"login:")
    telnet.write(username.encode() + b"\r\n")
    _ = telnet.read_until(b">")
except Exception as msg:
    print(msg)
    telnet.close()
    print("Połączenie nie powiodło się")
    exit()

print("Połączenie nawiązane")


def Command(cmd):
    telnet.write(cmd.encode()+b"\r\n")
    print(str(telnet.read_very_eager()))
    time.sleep(1)

Command("EXE PROJEKT_KOMP")

telnet.close()
print("Połączenie zamknięte")

