import socket

HOST = "192.168.4.1"
PORT = 3333

sock = socket.socket()
sock.connect((HOST, PORT))

while True:
    cmd = input("> ")

    sock.sendall((cmd + "\n").encode())

    try:
        sock.settimeout(2)
        resposta = sock.recv(1024)

        if resposta:
            print(resposta)
    except:
        pass