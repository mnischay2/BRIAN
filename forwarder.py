# combined_receiver_forwarder.py

import socket
import pyttsx3
import socketio
import random
import json
import time
import threading
from bs4 import BeautifulSoup
import requests
import subprocess

# === CONFIG ===
PORT_RESPONSE = 9000
SERVER_URL = "http://13.60.232.84:5003"
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjMGJmMGQ0OC04MWY2LTRmYjctYWMwOS02MjU2NjNhYjBlOTIiLCJ0SWQiOiI2YmE2NzA3NC0wNjE1LTQ4OWMtYmMwYy03OTEwYjA0MDYzYzIiLCJvcmdJZCI6ImUxNjhlMWU1LTEyMzEtNGE4ZC1hYmNmLWQ2NmM3NDU1MjIyOCIsImZpcnN0TmFtZSI6IklJUEQiLCJsYXN0TmFtZSI6ImJvZGhpIiwicm9sZSI6IlVTRVIiLCJzdWJzY3JpcHRpb25fcXVhbnRpdHkiOjgsImlhdCI6MTc1MjgzNjIzNiwiZXhwIjo0OTA4NTk2MjM2fQ.rce5Ayeaslhv5ZcjUX5mEc9MbgaVQzAhyOhn_reFDc0"

# === Speak Function ===
def speak_text(text):
    def tts_job():
        try:
            engine = pyttsx3.init()
            line=text.strip()
            print(f"🗣️ Speaking: {line}")
            engine.say(line)
            engine.runAndWait()  # Run the full queue at once
        except Exception as e:
            print(f"[!] TTS Error: {e}")

    threading.Thread(target=tts_job, daemon=True).start()


# === Session and HTTP ===
def generate_session_id():
    return f"yanshee_{random.randint(1000, 9999)}"

session_id = generate_session_id()
http_session = requests.Session()
http_session.headers.update({
    "X-Requested-With": "XMLHttpRequest"
})

# === Initialize Socket.IO client ===
sio = socketio.Client(
    http_session=http_session,
    logger=True,
    engineio_logger=True,
    reconnection=True
)

@sio.event
def connect():
    print("[✓] Connected to Socket.IO server.")
    print(f"[→] Session: {session_id}")

@sio.event
def connect_error(err):
    print("[!] Connection failed:", err)

@sio.on("response")
def on_response(data):
    print("[←] Server responded:")
    html_message = data.get("content", {}).get("message", "")
    soup = BeautifulSoup(html_message, "html.parser")
    plain_text = soup.get_text(separator="\n").strip()
    print(plain_text)

    # Speak text
    speak_text(plain_text)

@sio.on("error")
def on_error(data):
    print("[!] Error from server:")
    print(json.dumps(data, indent=2))

# === Receive & Forward ===
def receive_and_forward():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", PORT_RESPONSE))
    server.listen(1)
    print(f"📥 Listening for responses on port {PORT_RESPONSE}...")

    while True:
        conn, addr = server.accept()
        print(f"🧠 Message received from {addr}")
        buffer = b""
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buffer += data
        message = buffer.decode("utf-8").strip()

        if message:
            print(f"📨 Forwarding message to Socket.IO: {message}")
            payload = {
                "message_type": "user_message",
                "session_id": session_id,
                "bodhi_mode": "chat_with_expert_non_persistant",
                "content": {
                    "message": f"{message}. Summarize in two lines or less",
                    "session_name": "chatting with a knowledge expert"
                }
            }
            speak_text("Please wait while I generate an appropriate response")
            sio.emit("message", payload)

        conn.close()

# === Main Entry ===
if __name__ == "__main__":
    try:
        print(f"[→] Connecting to Socket.IO with session_id = {session_id}")
        sio.connect(
            SERVER_URL,
            transports=["polling", "websocket"],
            auth={
                "token": AUTH_TOKEN,
                "session_id": session_id
            },
            wait_timeout=20
        )

        receive_and_forward()

    except KeyboardInterrupt:
        print("\n[✗] Interrupted. Closing...")
        sio.emit("end_conversation", {
            "session_id": session_id,
            "bodhi_mode": "chat_with_expert_non_persistant"
        })
        sio.disconnect()

    except Exception as e:
        print("[✗] Exception:", e)