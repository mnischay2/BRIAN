#!/home/nischay/linenv/bin/python3
import socket
import struct
import numpy as np
import whisper
import torch
import time

# --- Whisper Model Configuration ---
# Automatically select GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Using device: {DEVICE}")
MODEL = whisper.load_model("medium.en", device=DEVICE)
print("[+] Whisper model large-v3 loaded.")

# --- Network Configuration ---
# Port to listen for mic client
MIC_LISTENER_HOST = "0.0.0.0"
MIC_LISTENER_PORT = 5001

# Address for the forwarder service
FORWARDER_HOST = "127.0.0.1"
FORWARDER_PORT = 5002

def connect_to_forwarder():
    """
    Attempts to connect to the forwarder service with retries.
    """
    while True:
        try:
            print(f"[*] Attempting to connect to forwarder at {FORWARDER_HOST}:{FORWARDER_PORT}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((FORWARDER_HOST, FORWARDER_PORT))
            print("[+] Connected to forwarder.")
            return sock
        except ConnectionRefusedError:
            print("[!] Forwarder connection refused. Is it running? Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"[!] Forwarder connection failed: {e}. Retrying in 5s...")
            time.sleep(5)

def send_to_forwarder(sock, text_data):
    """
    Sends transcribed text to the forwarder.
    Returns the same socket on success, or a newly reconnected socket on failure.
    """
    try:
        encoded_data = text_data.encode('utf-8')
        length = struct.pack('>I', len(encoded_data))
        sock.sendall(length)
        sock.sendall(encoded_data)
        return sock
    except (socket.error, BrokenPipeError, ConnectionResetError) as e:
        print(f"[!] Forwarder disconnected: {e}. Reconnecting...")
        sock.close()
        return connect_to_forwarder()

def handle_mic_client(conn, addr, forwarder_sock):
    """
    Handles a single connection from a microphone client.
    """
    print(f"[+] Mic client connected from {addr}")
    try:
        while True:
            # Receive the 4-byte length prefix
            length_bytes = conn.recv(4)
            if not length_bytes:
                print(f"[-] Mic client {addr} disconnected (no length).")
                break
            
            length = struct.unpack('>I', length_bytes)[0]

            # Receive the audio data
            data = b""
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    print(f"[-] Mic client {addr} disconnected (incomplete data).")
                    return forwarder_sock # Return current forwarder socket
                data += packet

            print(f"[*] Received {len(data)} bytes of audio data.")

            # --- Amplitude check removed as it's handled by the client ---
            
            # Process and transcribe the audio
            # Convert raw bytes to numpy array for transcription
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            # Use fp16 for faster inference if on CUDA
            audio_np = audio_int16.astype(np.float32) / 32768.0
            result = MODEL.transcribe(audio_np, language="en", fp16=torch.cuda.is_available())
            text = result['text'].strip()

            if text:
                print(f"📝 Transcription: {text}")
                forwarder_sock = send_to_forwarder(forwarder_sock, text)
            else:
                print("[-] Received empty transcription.")

    except ConnectionResetError:
        print(f"[-] Mic client {addr} disconnected forcefully.")
    except Exception as e:
        print(f"[!] Error handling client {addr}: {e}")
    finally:
        print(f"[*] Closing connection with {addr}.")
        conn.close()
    
    return forwarder_sock

def main():
    """
    Main server loop to listen for mic clients.
    """
    forwarder_sock = connect_to_forwarder()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((MIC_LISTENER_HOST, MIC_LISTENER_PORT))
        s.listen(1)
        print(f"[*] Transcription server listening on {MIC_LISTENER_HOST}:{MIC_LISTENER_PORT}")

        while True:
            try:
                conn, addr = s.accept()
                # Handle the client and update the forwarder socket in case it reconnected
                forwarder_sock = handle_mic_client(conn, addr, forwarder_sock)
            except KeyboardInterrupt:
                print("\n[!] Server shutting down by user request.")
                break
            except Exception as e:
                print(f"[!] An unexpected error occurred in the main loop: {e}")

    forwarder_sock.close()

if __name__ == "__main__":
    main()


