#!/home/nischay/linenv311/bin/python
import socket
import struct
import json
import os
import time
from datetime import datetime
import threading
import yaml
import sys

def load_config():
    """Loads the main configuration file."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[!!!] CRITICAL: config.yaml not found.")
        sys.exit(1)

class SessionManager:
    def __init__(self, config):
        self.log_dir = config['paths']['session_log_directory']
        self.timeout = config['session']['timeout_minutes'] * 60
        self.current_session = None
        self.session_file = None
        self.last_activity = None
        self.lock = threading.Lock()
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"[*] Created session log directory at: {self.log_dir}")

    def start_new_session(self):
        """Starts a new session, saving the previous one if it exists."""
        with self.lock:
            if self.current_session is not None:
                self.save_session()
            
            session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_file = os.path.join(self.log_dir, f"session_{session_id}.json")
            self.current_session = []
            self.last_activity = time.time()
            print(f"[*] Starting new session: {self.session_file}")

    def add_entry(self, entry_data):
        """Adds a new interaction to the current session."""
        with self.lock:
            if self.current_session is None:
                self.start_new_session()
            
            self.current_session.append({
                "timestamp": datetime.now().isoformat(),
                "interaction": entry_data
            })
            self.last_activity = time.time()
            self.save_session()
            print(f"[+] Added entry to session {os.path.basename(self.session_file)}.")

    def save_session(self):
        """Saves the current session data to its JSON file."""
        if self.session_file and self.current_session is not None:
            try:
                with open(self.session_file, 'w') as f:
                    json.dump(self.current_session, f, indent=4)
            except Exception as e:
                print(f"[!] Error saving session file: {e}")

    def check_timeout(self):
        """Periodically checks if the session has timed out due to inactivity."""
        while True:
            time.sleep(60) # Check every minute
            with self.lock:
                if self.current_session is not None and (time.time() - self.last_activity > self.timeout):
                    print(f"[*] Session timed out. Saving and closing session file.")
                    self.save_session()
                    self.current_session = None
                    self.session_file = None

def handle_client(conn, manager):
    """Handles the incoming connection from the central service."""
    print("[+] Central service connected to session manager.")
    try:
        with conn:
            while True:
                # 1. Read the 4-byte length prefix
                length_bytes = conn.recv(4)
                if not length_bytes:
                    break
                length = struct.unpack('>I', length_bytes)[0]
                
                # 2. Loop to ensure the full message is received
                data = b""
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet:
                        break
                    data += packet
                
                if len(data) < length:
                    break # Incomplete message received

                # 3. Process the complete message
                try:
                    interaction = json.loads(data.decode('utf-8'))
                    manager.add_entry(interaction)
                except json.JSONDecodeError as e:
                    print(f"[!] Received malformed JSON data: {e}")

    except (ConnectionResetError, BrokenPipeError):
        print("[-] Central service disconnected from session manager.")
    finally:
        print("[*] Session manager client handler finished.")

def main():
    config = load_config()
    try:
        host = config['ports']['session_manager']['host']
        port = config['ports']['session_manager']['port']
    except KeyError as e:
        print(f"[!!!] CRITICAL: Missing configuration in config.yaml for session_manager. Key not found: {e}")
        sys.exit(1)

    manager = SessionManager(config)
    manager.start_new_session()
    
    threading.Thread(target=manager.check_timeout, daemon=True).start()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # --- Add retry logic for binding the socket ---
    for i in range(10): # Retry for 10 seconds
        try:
            server_socket.bind((host, port))
            server_socket.listen()
            print(f"[*] Session Manager listening on {host}:{port}")
            break
        except OSError as e:
            if e.errno == 98: # Address already in use
                print(f"[!] Port {port} is in use, retrying... ({i+1}/10)")
                time.sleep(1)
            else:
                print(f"[!!!] An unexpected error occurred while binding: {e}")
                sys.exit(1)
    else: # This else belongs to the for loop; it runs if the loop completes without a break
        print(f"[!!!] Failed to bind to port {port} after multiple retries. Exiting.")
        server_socket.close()
        sys.exit(1)

    try:
        while True:
            conn, _ = server_socket.accept()
            # Handle each client in a new thread
            threading.Thread(target=handle_client, args=(conn, manager), daemon=True).start()
    except KeyboardInterrupt:
        print("\n[*] Shutting down session manager.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()

