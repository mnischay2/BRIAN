#!/home/nischay/linenv311/bin/python
import socket
import struct
import numpy as np
import torch
import time
import yaml
import sys
from whisper import load_model
import port_config as pc_


def main():
    
    model_name = "large-v3"
    mic_host = "0.0.0.0"
    mic_port = pc_.transcriber
    central_host = "127.0.0.1"
    central_port = pc_.central
    # stt device (cpu/gpu)
    stt_device_pref = "gpu"

    # choose device based on config and availability
    if stt_device_pref.lower() == 'gpu' and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
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

                print(f"[*] Received {len(data)} bytes of audio data.")

                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                result = model.transcribe(audio_np, language="en", fp16=(device=="cuda"))
                text = result.get('text', '').strip()

                if text:
                    print(f"📝 Transcription: {text}")
                    central_sock = send_to_central(central_sock, text, central_host, central_port)

    except (ConnectionResetError, BrokenPipeError):
        print(f"[-] Mic client {addr} disconnected.")
    finally:
        print(f"[-] Connection closed for mic client {addr}")
    return central_sock

def send_to_central(sock, text, host, port):
    try:
        encoded_text = text.encode('utf-8')
        length = struct.pack('>I', len(encoded_text))
        sock.sendall(length)
        sock.sendall(encoded_text)
        return sock
    except (socket.error, BrokenPipeError):
        print("[!] Central service disconnected. Reconnecting...")
        sock.close()
        return connect_to_central(host, port)

if __name__ == "__main__":
    main()
