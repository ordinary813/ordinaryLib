import socket

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        msg = input()
        if msg == 'exit':
            print("Connection terminating...")
            break
        s.sendall(msg.encode())
        data = s.recv(1024)
        print(f'Sent: {data.decode()}')