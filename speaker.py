#!/home/nischay/linenv311/bin/python
import socket
import threading
import queue
import struct
import time
import pyttsx3
import yaml
import sys

# --- Global State ---
speaker_status = "IDLE"
status_lock = threading.Lock()
text_queue = queue.Queue()

def load_config():
    """Loads the main configuration file."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[!!!] CRITICAL: config.yaml not found. Speaker service cannot start.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[!!!] CRITICAL: Error parsing config.yaml: {e}")
        sys.exit(1)

def tts_worker(config):
    """
    A worker thread that takes text from a queue, speaks it using pyttsx3,
    and manages the global 'speaker_status'.
    """
    global speaker_status
    
    try:
        engine = pyttsx3.init()
        
        # Load settings from the config file
        rate = config['tts']['pyttsx3']['rate']
        voice_index = config['tts']['pyttsx3']['voice_index']
        
        engine.setProperty('rate', rate)
        
        voices = engine.getProperty('voices')
        
        if 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
            print(f"[*] TTS engine initialized with voice: {voices[voice_index].name} (Rate: {rate})")
        else:
            print(f"[!] Warning: Voice index {voice_index} is out of range. Using default voice.")

    except Exception as e:
        print(f"[!!!] Failed to initialize pyttsx3 engine: {e}")
        return # Exit the thread if the engine fails

    while True:
        text_to_speak = text_queue.get()
        
        with status_lock:
            speaker_status = "BUSY"
        print(f"[*] Speaking: {text_to_speak}")
        
        try:
            engine.say(text_to_speak)
            engine.runAndWait()
        except Exception as e:
            print(f"[!] An error occurred in the TTS worker: {e}")
        finally:
            # --- Dynamic Sleep Calculation ---
            # Calculate a dynamic delay based on the text length to ensure the audio
            # buffer clears before setting the status to IDLE. This prevents the mic
            # from starting while the last word is still playing.
            # Formula: A base delay + a small fraction of time per character.
            base_delay = 0.2  # Minimum 200ms delay
            per_char_delay = 0.005  # 5 milliseconds per character
            dynamic_delay = base_delay + (len(text_to_speak) * per_char_delay)
            
            # Cap the delay to a maximum reasonable value (e.g., 2 seconds)
            final_delay = min(dynamic_delay, 2.0)

            print(f"[*] Using dynamic sleep time: {final_delay:.2f}s")
            time.sleep(final_delay)
            
            with status_lock:
                speaker_status = "IDLE"
            print("[*] Finished speaking. Status is now IDLE.")
            text_queue.task_done()

def handle_connection(conn, addr, name="Client"):
    """Handles a connection from the central service."""
    print(f"[+] {name} connected from {addr}")
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
                print(f"[*] Received text to speak: '{text}'")
                text_queue.put(text)
    except ConnectionResetError:
        print(f"[-] {name} at {addr} disconnected.")
    finally:
        print(f"[-] Connection closed for {addr}")
        conn.close()

def start_server(host, port, handler, handler_args=()):
    """Generic server starter that listens on a port and handles connections."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Retry mechanism for binding to the port
    for i in range(10):
        try:
            server_socket.bind((host, port))
            server_socket.listen()
            print(f"[*] Server listening on {host}:{port}")
            break
        except OSError as e:
            if e.errno == 98:
                print(f"[!] Port {port} is in use, retrying... ({i+1}/10)")
                time.sleep(1)
            else:
                print(f"[!!!] Failed to bind to {host}:{port}: {e}")
                return
    else:
        print(f"[!!!] Failed to bind to port {port} after multiple retries. Exiting.")
        server_socket.close()
        return

    while True:
        conn, addr = server_socket.accept()
        handler(conn, addr, *handler_args)

def status_server_handler(conn, addr):
    """Special handler for the status server; sends status and closes."""
    with conn:
        with status_lock:
            current_status = speaker_status
        conn.sendall(current_status.encode('utf-8'))

if __name__ == "__main__":
    config = load_config()
    
    # --- Get port configurations with validation ---
    try:
        speaker_config = config['ports']['speaker']
        text_host = speaker_config['text_host']
        text_port = speaker_config['text_port']
        status_host = speaker_config['status_host']
        status_port = speaker_config['status_port']
    except KeyError as e:
        print(f"[!!!] CRITICAL: Missing configuration in config.yaml. Could not find key: {e}")
        print("[!!!] Please ensure your config.yaml has a 'ports' section with a 'speaker' subsection containing all required hosts and ports.")
        sys.exit(1)

    print("[*] Starting Speaker Service (using pyttsx3)...")
    
    threading.Thread(target=tts_worker, args=(config,), daemon=True).start()
    threading.Thread(target=start_server, args=(status_host, status_port, status_server_handler), daemon=True).start()
    
    start_server(text_host, text_port, handle_connection, handler_args=("Central service",))

