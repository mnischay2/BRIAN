#!/home/nischay/linenv311/bin/python
import socket
import threading
from gtts import gTTS
import queue
import struct
import time
import os
import subprocess

# --- Configuration ---
TEXT_LISTENER_HOST = "0.0.0.0"
TEXT_LISTENER_PORT = 5003

STATUS_HOST = "0.0.0.0"
STATUS_PORT = 5004

# --- Global State ---
speaker_status = "IDLE"
status_lock = threading.Lock()
text_queue = queue.Queue()

def tts_worker():
    """
    A worker thread that takes text from a queue, generates an MP3 using gTTS,
    plays it, and manages the global 'speaker_status'.
    """
    global speaker_status
    tts_file = "brian_tts_output.mp3"
    
    while True:
        text_to_speak = text_queue.get()
        
        with status_lock:
            speaker_status = "BUSY"
        print(f"[*] Generating speech for: {text_to_speak}")
        
        try:
            # 1. Generate speech using gTTS with the Indian English accent
            tts = gTTS(text=text_to_speak, lang='en', tld='co.in')
            tts.save(tts_file)
            
            # 2. Play the generated audio file using mpg123
            # The subprocess call is blocking, so it will wait until playback is finished.
            # Using -q for quiet mode to prevent mpg123 from printing to the console.
            subprocess.call(["mpg123", "-q", tts_file])
            
        except Exception as e:
            print(f"[!] An error occurred in the TTS worker: {e}")
        finally:
            # 3. Clean up the temporary audio file
            if os.path.exists(tts_file):
                os.remove(tts_file)
            
            with status_lock:
                speaker_status = "IDLE"
            print("[*] Finished speaking. Status is now IDLE.")
            text_queue.task_done()

def handle_forwarder_client(conn, addr):
    """
    Handles a connection from the forwarder, expecting length-prefixed messages.
    """
    print(f"[+] Forwarder connected from {addr}")
    try:
        while True:
            length_bytes = conn.recv(4)
            if not length_bytes: break
            length = struct.unpack('>I', length_bytes)[0]
            
            data = b""
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet: break
                data += packet
            
            if data:
                text = data.decode('utf-8')
                text_queue.put(text)

    except ConnectionResetError:
        print(f"[-] Forwarder at {addr} disconnected abruptly.")
    finally:
        print(f"[-] Connection closed for forwarder {addr}")
        conn.close()

def text_listener_server():
    """Listens for connections from forwarder.py to receive text."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TEXT_LISTENER_HOST, TEXT_LISTENER_PORT))
        s.listen()
        print(f"[*] Speaker text listener running on {TEXT_LISTENER_HOST}:{TEXT_LISTENER_PORT}")
        while True:
            conn, addr = s.accept()
            handle_forwarder_client(conn, addr)

def status_server():
    """Listens for connections from mic.py to report status."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((STATUS_HOST, STATUS_PORT))
        s.listen()
        print(f"[*] Speaker status server running on {STATUS_HOST}:{STATUS_PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                with status_lock:
                    current_status = speaker_status
                conn.sendall(current_status.encode('utf-8'))

if __name__ == "__main__":
    print("[*] Starting Speaker Service...")
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=status_server, daemon=True).start()
    text_listener_server()

