#!/home/nischay/linenv311/bin/python
import socket
import struct
import tkinter as tk
from tkinter import scrolledtext, font as tkfont
import threading
import queue
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

class AssistantUI:
    def __init__(self, root):
        self.root = root
        self.root.title("B.R.I.A.N. - Control Panel")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")
        self.message_queue = queue.Queue()

        self._setup_fonts()
        self._setup_ui()
        
        self.root.after(100, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_fonts(self):
        self.default_font = tkfont.Font(family="Roboto", size=12)
        self.status_font = tkfont.Font(family="Roboto", size=10, weight="bold")
        self.title_font = tkfont.Font(family="Roboto", size=16, weight="bold")

    def _setup_ui(self):
        title = tk.Label(self.root, text="B.R.I.A.N.", font=self.title_font, bg="#1e1e1e", fg="#00aaff")
        title.pack(pady=(10, 5))

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled',
                                                 font=self.default_font, bg="#2a2a2a", fg="#e0e0e0",
                                                 insertbackground='white', bd=0, relief=tk.FLAT)
        self.text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        status_frame = tk.Frame(self.root, bg="#1e1e1e")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.wake_status = tk.Label(status_frame, text="WAKE: UNKNOWN", font=self.status_font, bg="#1e1e1e")
        self.wake_status.pack(side=tk.LEFT, padx=(5, 20))
        self.update_status(self.wake_status, "WAKE", "SLEEPING", "#e07b7b")

        self.llm_status = tk.Label(status_frame, text="LLM: UNKNOWN", font=self.status_font, bg="#1e1e1e")
        self.llm_status.pack(side=tk.LEFT)
        self.update_status(self.llm_status, "LLM", "IDLE", "#a0a0a0")

    def update_status(self, label, prefix, text, color):
        label.config(text=f"{prefix}: {text}", fg=color)

    def update_text_area(self, user, message):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, f"{user}: {message}\n\n")
        self.text_area.config(state='disabled')
        self.text_area.yview(tk.END)

    def process_queue(self):
        try:
            message = self.message_queue.get_nowait()
            print(f"[UI DEBUG] Processing message from queue: '{message}'") # Debug line added
            parts = message.split(':', 1)
            msg_type = parts[0]
            content = parts[1] if len(parts) > 1 else ""

            if msg_type == "wake_status":
                color = "#7be08a" if content == "LISTENING" else "#e07b7b"
                self.update_status(self.wake_status, "WAKE", content, color)
            elif msg_type == "llm_status":
                colors = {"IDLE": "#a0a0a0", "THINKING": "#e0d37b", "SPEAKING": "#7bcee0"}
                self.update_status(self.llm_status, "LLM", content, colors.get(content, "#a0a0a0"))
            elif msg_type == "user_transcription":
                self.update_text_area("You", content)
            elif msg_type == "llm_response":
                self.update_text_area("Assistant", content)
            elif msg_type == "system_message":
                self.update_text_area("System", content)

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)
    
    def on_closing(self):
        self.root.destroy()

def handle_central_client(conn, msg_queue):
    """Handles the connection from the central service, receiving all UI updates."""
    print("[+] Central service connected to UI.")
    try:
        with conn:
            while True:
                # 1. Read the 4-byte length prefix to determine message size
                length_bytes = conn.recv(4)
                if not length_bytes:
                    print("[-] Central service closed the connection (no length).")
                    break
                length = struct.unpack('>I', length_bytes)[0]
                
                # 2. Loop to ensure the full message is received
                data = b""
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet:
                        print("[-] Central service closed the connection (incomplete data).")
                        break
                    data += packet
                
                # This outer break is necessary if the inner loop breaks
                if len(data) < length:
                    break

                # 3. Put the complete message into the queue for the GUI thread
                msg_queue.put(data.decode('utf-8'))

    except (ConnectionResetError, BrokenPipeError):
        print("[-] Central service disconnected from UI.")
    finally:
        print("[*] UI client handler finished.")

def run_server(host, port, msg_queue):
    """
    The main server loop for the UI. It listens for a single connection
    from the central service and passes it to the handler.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"[*] UI Server listening for Central on {host}:{port}")
        while True:
            # This server is designed to only handle one client: the central service.
            # If the client disconnects, the handler will finish, and this loop
            # will wait to accept a new connection.
            conn, addr = s.accept()
            handle_central_client(conn, msg_queue)

def main():
    config = load_config()
    try:
        host = config['ports']['ui']['host']
        port = config['ports']['ui']['port']
    except KeyError as e:
        print(f"[!!!] CRITICAL: Missing configuration in config.yaml. Key not found: {e}")
        sys.exit(1)

    root = tk.Tk()
    app = AssistantUI(root)
    
    # Start the network listener in a separate thread so it doesn't block the GUI
    server_thread = threading.Thread(target=run_server, args=(host, port, app.message_queue), daemon=True)
    server_thread.start()
    
    root.mainloop()

if __name__ == "__main__":
    main()

