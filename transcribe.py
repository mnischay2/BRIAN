#!/home/nischay/linenv311/bin/python
import socket
import struct
import numpy as np
import torch
import time
import yaml
import sys
from whisper import load_model

def load_config():
    """Loads the main configuration file."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[!!!] CRITICAL: config.yaml not found.")
        sys.exit(1)

def main():
    config = load_config()

    # --- Load Configuration ---
    try:
        model_name = config['models']['whisper']
        transcriber_config = config['ports']['transcriber']
        mic_host = transcriber_config['mic_host']
        mic_port = transcriber_config['mic_port']
        central_host = transcriber_config['central_host']
        central_port = transcriber_config['central_port']
    except KeyError as e:
        print(f"[!!!] CRITICAL: Missing configuration in config.yaml. Key not found: {e}")
        sys.exit(1)

    # --- Whisper Model Initialization ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")
    model = load_model(model_name, device=device)
    print(f"[+] Whisper model '{model_name}' loaded.")

    # --- Main Server Loop ---
    central_sock = connect_to_central(central_host, central_port)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((mic_host, mic_port))
        s.listen()
        print(f"[*] Transcriber listening for mic on {mic_host}:{mic_port}")
        
        while True:
            conn, addr = s.accept()
            # Since we only expect one mic, we handle it in the main thread.
            # For multiple mics, a new thread would be needed here.
            central_sock = handle_mic_client(conn, addr, model, central_sock, central_host, central_port, device)

def connect_to_central(host, port):
    """Connects to the central service with retries."""
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
    """Handles a connection from the mic, transcribes, and forwards."""
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
                text = result['text'].strip()

                if text:
                    print(f"📝 Transcription: {text}")
                    central_sock = send_to_central(central_sock, text, central_host, central_port)

    except (ConnectionResetError, BrokenPipeError):
        print(f"[-] Mic client {addr} disconnected.")
    finally:
        print(f"[-] Connection closed for mic client {addr}")
    return central_sock

def send_to_central(sock, text, host, port):
    """Sends the transcribed text to the central service."""
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

