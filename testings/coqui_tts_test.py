# client.py
import socket

HOST = "127.0.0.1"
PORT = 5005

while True:
    text = input("Enter text (or 'exit' to quit): ")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(text.encode("utf-8"))
        if text.lower() == "exit":
            break
