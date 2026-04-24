import socket
import struct
import numpy as np
import torch
import time
import sys
import json
from whisper import load_model
import scripts.configs.port_config as pc_

def main():
    model_name = "large-v3"
    mic_host = "0.0.0.0"
    mic_port = pc_.transcriber
    central_host = "127.0.0.1"
    central_port = pc_.central
    stt_device_pref = "gpu"

    device = "cuda" if stt_device_pref.lower() == 'gpu' and torch.cuda.is_available() else "cpu"
    print(f"[*] Using device for STT: {device}")

    model = load_model(model_name, device=device)
    print(f"[+] Whisper model '{model_name}' loaded on {device}.")

    central_sock = connect_to_central(central_host, central_port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((mic_host, mic_port))
        s.listen()
        print(f"[*] Transcriber listening for mic on {mic_host}:{mic_port}")

        while True:
            conn, addr = s.accept()
            central_sock = handle_mic_client(conn, addr, model, central_sock, central_host, central_port, device)

def connect_to_central(host, port):
    while True:
        try:
            print(f"[*] Transcriber connecting to central service at {host}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print("[+] Transcriber connected to central service.")
            return sock
        except Exception as e:
            print(f"[!] Connection to central failed: {e}. Retrying in 5s...")
            time.sleep(5)

def handle_mic_client(conn, addr, model, central_sock, central_host, central_port, device):
    print(f"[+] Mic client connected from {addr}")
    try:
        with conn:
            while True:
                length_bytes = conn.recv(4)
                if not length_bytes: break

                length = struct.unpack('>I', length_bytes)[0]
                data = b""
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet: break
                    data += packet

                if not data: break

                # --- MODIFICATION START ---
                # Inform central that we are processing audio
                print("[*] Received audio, notifying central of 'Listening' state.")
                central_sock = send_to_central(central_sock, {"type": "status", "payload": "Listening"}, central_host, central_port)

                # Transcribe the audio
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                result = model.transcribe(audio_np, language="en", fp16=(device=="cuda"))
                text = result.get('text', '').strip()

                if text:
                    print(f"📝 Transcription: {text}")
                    # Send the final transcript
                    central_sock = send_to_central(central_sock, {"type": "transcript", "payload": text}, central_host, central_port)
                # --- MODIFICATION END ---

    except (ConnectionResetError, BrokenPipeError):
        print(f"[-] Mic client {addr} disconnected.")
    finally:
        print(f"[-] Connection closed for mic client {addr}")
    return central_sock

def send_to_central(sock, data, host, port):
    """Sends a dictionary as a JSON payload to the central server."""
    try:
        payload = json.dumps(data).encode('utf-8')
        length = struct.pack('>I', len(payload))
        sock.sendall(length + payload)
        return sock
    except (socket.error, BrokenPipeError):
        print("[!] Central service disconnected. Reconnecting...")
        sock.close()
        return connect_to_central(host, port)

if __name__ == "__main__":
    main()
