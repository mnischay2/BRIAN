#!/home/nischay/linenv311/bin/python
import socket
import struct
import threading
import requests
import json
import time
import yaml
import sys
import re

def load_config():
    """Loads the main configuration file."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[!!!] CRITICAL: config.yaml not found.")
        sys.exit(1)

class CentralOrchestrator:
    def __init__(self, config):
        self.config = config
        self.is_awake = False
        
        # --- Load Configuration with Validation ---
        try:
            self.wake_words = self.config['wake_words']
            
            central_ports = self.config['ports']['central']
            self.transcriber_listen_host = central_ports['transcriber_host']
            self.transcriber_listen_port = central_ports['transcriber_port']
            self.speaker_connect_host = central_ports['speaker_host']
            self.speaker_connect_port = central_ports['speaker_port']
            self.session_connect_host = central_ports['session_host']
            self.session_connect_port = central_ports['session_port']
            self.ui_connect_host = central_ports['ui_host']
            self.ui_connect_port = central_ports['ui_port']

            model_config = self.config['models']
            self.ollama_model = model_config['ollama']
            self.ollama_endpoint = model_config['ollama_endpoint']

        except KeyError as e:
            print(f"[!!!] CRITICAL: Missing configuration in config.yaml. Key not found: {e}")
            sys.exit(1)

        # Sockets
        self.speaker_sock = self.connect_to_service("Speaker", self.speaker_connect_host, self.speaker_connect_port)
        self.session_sock = self.connect_to_service("Session Manager", self.session_connect_host, self.session_connect_port)
        self.ui_sock = self.connect_to_service("UI", self.ui_connect_host, self.ui_connect_port)

    def connect_to_service(self, name, host, port):
        """Generic connection function with retries."""
        while True:
            try:
                print(f"[*] Central connecting to {name} at {host}:{port}...")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                print(f"[+] Central connected to {name}.")
                return sock
            except Exception as e:
                print(f"[!] Connection to {name} failed: {e}. Retrying...")
                time.sleep(3)

    def send_with_reconnect(self, sock, service_name, host, port, data):
        """Sends data and handles reconnection on failure."""
        try:
            sock.sendall(data)
            return sock
        except (socket.error, BrokenPipeError):
            print(f"[!] {service_name} disconnected. Reconnecting...")
            sock.close()
            new_sock = self.connect_to_service(service_name, host, port)
            new_sock.sendall(data)
            return new_sock

    def send_length_prefixed(self, service_name, text):
        """Encodes text and sends it with a 4-byte length prefix to the correct service."""
        encoded_text = text.encode('utf-8')
        length_prefix = struct.pack('>I', len(encoded_text))
        data_to_send = length_prefix + encoded_text
        
        if service_name == "Speaker":
            self.speaker_sock = self.send_with_reconnect(self.speaker_sock, "Speaker", self.speaker_connect_host, self.speaker_connect_port, data_to_send)
        elif service_name == "Session Manager":
            self.session_sock = self.send_with_reconnect(self.session_sock, "Session Manager", self.session_connect_host, self.session_connect_port, data_to_send)
        elif service_name == "UI":
            self.ui_sock = self.send_with_reconnect(self.ui_sock, "UI", self.ui_connect_host, self.ui_connect_port, data_to_send)

    def clean_text_for_speech(self, text):
        """
        Removes symbols that are poorly handled by TTS, while keeping
        essential punctuation for natural speech flow.
        """
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,?!']", " ", text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def llm_worker(self, command_text):
        """Handles the interaction with the Ollama LLM."""
        try:
            self.send_length_prefixed("UI", f"llm_status:THINKING")
            
            payload = {"model": self.ollama_model, "prompt": command_text, "stream": False}
            response = requests.post(self.ollama_endpoint, json=payload, timeout=60)
            response.raise_for_status()
            
            llm_response = json.loads(response.text).get("response", "I'm sorry, I encountered an error.").strip()
            
            self.send_length_prefixed("UI", f"llm_response:{llm_response}")
            session_data = json.dumps({"question": command_text, "answer": llm_response})
            self.send_length_prefixed("Session Manager", session_data)

            speech_text = self.clean_text_for_speech(llm_response)
            self.send_length_prefixed("UI", f"llm_status:SPEAKING")
            self.send_length_prefixed("Speaker", speech_text)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to LLM: {e}"
            self.send_length_prefixed("UI", f"system_message:{error_msg}")
        finally:
            self.is_awake = False
            self.send_length_prefixed("UI", "wake_status:SLEEPING")
            self.send_length_prefixed("UI", "llm_status:IDLE")

    def process_transcription(self, text):
        """Processes transcribed text to check for wake words or commands."""
        print(f"[*] Processing transcription: '{text}' (Awake state: {self.is_awake})")
        self.send_length_prefixed("UI", f"user_transcription:{text}")
        
        if self.is_awake:
            print("[*] Assistant is awake. Treating as a command.")
            threading.Thread(target=self.llm_worker, args=(text,)).start()
        else:
            print("[*] Assistant is sleeping. Checking for wake word...")
            if any(word in text for word in self.wake_words):
                print("[+] Wake word detected! Setting state to AWAKE and LISTENING.")
                self.is_awake = True
                self.send_length_prefixed("UI", "wake_status:LISTENING")
            else:
                print("[-] No wake word detected.")

    def handle_transcriber_client(self, conn):
        """Receives data from the transcriber service."""
        print("[+] Transcriber client connected.")
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
                    if data:
                        self.process_transcription(data.decode('utf-8'))
        except (ConnectionResetError, BrokenPipeError):
            print("[-] Transcriber client disconnected.")

    def start(self):
        """Starts the main listener for the transcriber service."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # --- Add retry logic for binding the socket ---
        for i in range(10): # Retry for 10 seconds
            try:
                server_socket.bind((self.transcriber_listen_host, self.transcriber_listen_port))
                server_socket.listen()
                print(f"[*] Central service listening for transcriber on {self.transcriber_listen_host}:{self.transcriber_listen_port}")
                break
            except OSError as e:
                if e.errno == 98: # Address already in use
                    print(f"[!] Port {self.transcriber_listen_port} is in use, retrying... ({i+1}/10)")
                    time.sleep(1)
                else:
                    print(f"[!!!] An unexpected error occurred while binding: {e}")
                    sys.exit(1)
        else: # This else belongs to the for loop; it runs if the loop completes without a break
            print(f"[!!!] Failed to bind to port {self.transcriber_listen_port} after multiple retries. Exiting.")
            server_socket.close()
            sys.exit(1)

        try:
            while True:
                conn, _ = server_socket.accept()
                threading.Thread(target=self.handle_transcriber_client, args=(conn,), daemon=True).start()
        except KeyboardInterrupt:
            print("\n[*] Shutting down central service.")
        finally:
            server_socket.close()

if __name__ == "__main__":
    config = load_config()
    orchestrator = CentralOrchestrator(config)
    orchestrator.start()

