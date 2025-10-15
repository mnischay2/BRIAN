import socket
import threading
import json
import struct
import tkinter as tk
from tkinter import scrolledtext, PhotoImage
from queue import Queue, Empty
import port_config as pc_

# --- Configuration ---
CENTRAL_HOST = "127.0.0.1"
CENTRAL_PORT = pc_.ui_client

class ChatClient:
    def __init__(self, root):
        self.root = root
        self.root.title("Brian")
        self.root.geometry("500x650")

        # --- Style Configuration ---
        self.BG_COLOR = "#1e1e1e"
        self.TEXT_COLOR = "#dcdcdc"
        self.USER_COLOR = "#6495ed"  # Cornflower Blue
        self.ASSISTANT_COLOR = "#98fb98"  # Pale Green
        self.STATUS_COLOR = "#b0c4de"  # Light Steel Blue
        self.FONT_FAMILY = "Segoe UI"
        
        self.root.configure(bg=self.BG_COLOR)

        # --- Message Queue for Thread-Safe UI Updates ---
        self.queue = Queue()

        # --- UI Elements ---
        self.setup_ui()
        
        # --- Network Communication ---
        self.sock = None
        self.connect_to_server()

        # --- Start processing messages from the queue ---
        self.process_queue()

    def setup_ui(self):
        # --- Chat Display ---
        self.chat_display = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state='disabled',
            bg="#252526", fg=self.TEXT_COLOR,
            font=(self.FONT_FAMILY, 11), relief=tk.FLAT,
            padx=10, pady=10
        )
        self.chat_display.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # --- Tag configurations for user and assistant ---
        self.chat_display.tag_config('user', foreground=self.USER_COLOR, font=(self.FONT_FAMILY, 11, 'bold'))
        self.chat_display.tag_config('assistant', foreground=self.ASSISTANT_COLOR)
        self.chat_display.tag_config('info', foreground=self.STATUS_COLOR, justify='center', font=(self.FONT_FAMILY, 9, 'italic'))

        # --- Status Bar ---
        self.status_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_icon = tk.Label(self.status_frame, text="●", font=(self.FONT_FAMILY, 14), bg=self.BG_COLOR, fg="grey")
        self.status_icon.pack(side=tk.LEFT, padx=(0, 5))
        
        self.status_label = tk.Label(
            self.status_frame, text="Connecting...", anchor='w',
            bg=self.BG_COLOR, fg=self.STATUS_COLOR,
            font=(self.FONT_FAMILY, 10)
        )
        self.status_label.pack(side=tk.LEFT)

    def connect_to_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((CENTRAL_HOST, CENTRAL_PORT))
            self.update_status("Connected", "green")

            # Start a thread to listen for server messages
            receive_thread = threading.Thread(target=self.receive_messages, daemon=True)
            receive_thread.start()
        except ConnectionRefusedError:
            self.update_status("Connection Failed", "red")
            self.add_message("System", "Could not connect to the central server. Is it running?")

    def receive_messages(self):
        while True:
            try:
                length_bytes = self.sock.recv(4)
                if not length_bytes:
                    break
                
                length = struct.unpack('>I', length_bytes)[0]
                data = b""
                while len(data) < length:
                    packet = self.sock.recv(length - len(data))
                    if not packet:
                        break
                    data += packet

                if data:
                    message = json.loads(data.decode('utf-8'))
                    self.queue.put(message)

            except (ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                print(f"[UI Error] {e}")
                break
        
        self.queue.put({"type": "status", "payload": {"status": "Disconnected", "color": "red"}})

    def process_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                msg_type = message.get("type")

                if msg_type == "status":
                    payload = message.get("payload", {})
                    self.update_status(payload.get("status", "Unknown"), payload.get("color", "grey"))
                elif msg_type == "chat":
                    payload = message.get("payload", {})
                    role = payload.get("role")
                    content = payload.get("content")
                    if role and content:
                        self.add_message(role, content)
                elif msg_type == "clear":
                    self.clear_chat()

        except Empty:
            pass  # No new messages
        finally:
            self.root.after(100, self.process_queue)

    def add_message(self, role, content):
        self.chat_display.config(state='normal')
        
        role_label = "You: " if role.lower() == 'user' else "Brian: "
        
        self.chat_display.insert(tk.END, role_label, role.lower())
        self.chat_display.insert(tk.END, content + "\n\n")
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def update_status(self, status_text, color):
        self.status_label.config(text=status_text)
        self.status_icon.config(fg=color)

    def clear_chat(self):
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state='disabled')

    def on_closing(self):
        if self.sock:
            self.sock.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    client = ChatClient(root)
    root.protocol("WM_DELETE_WINDOW", client.on_closing)
    root.mainloop()
