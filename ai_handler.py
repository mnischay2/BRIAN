# ai_handler.py
import socket
import threading
import requests
import json
import port_config as pc_

HOST = "127.0.0.1"
PORT = pc_.ai_handler

OLLAMA_URL = "http://localhost:11434/api/chat"

def handle_client(conn, addr):
    print(f"[+] Connected by {addr}")
    try:
        # Receive user message
        prompt = conn.recv(4096).decode().strip()
        if not prompt:
            conn.close()
            return

        print(f"[User Prompt] {prompt}")

        payload = {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": "You are an assistant named BRIAN. Answer briefly and naturally no markup or aserisks."},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        conn.sendall(token.encode("utf-8"))
                    if data.get("done"):
                        break

        conn.sendall(b"\n--- done ---\n")
    except Exception as e:
        print(f"[!] Error: {e}")
        conn.sendall(f"\n[Error: {e}]".encode("utf-8"))
    finally:
        conn.close()
        print(f"[-] Disconnected {addr}")

def start_server():
    print(f"[*] Starting AI server on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
