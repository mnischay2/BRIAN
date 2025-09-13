
#!/home/nischay/linenv/bin/python3
import socket
import tkinter as tk
from tkinter import scrolledtext, font
import threading
import queue
import time
import struct

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 5002

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Transcription")
        self.root.geometry("700x450")
        self.root.configure(bg="#2E2E2E")

        # --- Fonts ---
        self.main_font = font.Font(family="Helvetica", size=12)
        self.status_font = font.Font(family="Helvetica", size=10)

        # --- Main Text Area ---
        self.text_area = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            state='disabled',
            font=self.main_font,
            bg="#1E1E1E",
            fg="#D4D4D4",
            insertbackground="#FFFFFF"
        )
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(
            root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=self.status_font,
            bg="#007ACC",
            fg="#FFFFFF"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var.set("  Initializing...")

        # --- Communication Queue ---
        self.message_queue = queue.Queue()
        self.is_running = True

        # --- Start Server Thread ---
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()

        # --- GUI Update Loop ---
        self.root.after(100, self.process_queue)

        # --- Graceful Shutdown ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_server(self):
        """ The main server loop that listens for and handles connections. """
        self.status_var.set("  Listening on {}:{}".format(HOST, PORT))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((HOST, PORT))
                s.listen()
            except Exception as e:
                self.status_var.set(f"  Error: {e}")
                return

            while self.is_running:
                try:
                    conn, addr = s.accept()
                    client_thread = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                    client_thread.start()
                except OSError:
                    # This can happen when the socket is closed during shutdown.
                    break
                except Exception as e:
                    print(f"[!] Server accept error: {e}")
                    time.sleep(1)

    def handle_client(self, conn, addr):
        """ Handles a single client connection in its own thread using a length-prefix protocol. """
        print(f"[+] Client connected from {addr}")
        self.status_var.set(f"  Client connected from {addr[0]}")

        try:
            with conn:
                while self.is_running:
                    # 1. Receive the 4-byte length prefix
                    length_bytes = conn.recv(4)
                    if not length_bytes:
                        break  # Connection closed by client

                    # 2. Unpack the length (big-endian unsigned integer)
                    length = struct.unpack('>I', length_bytes)[0]

                    # 3. Receive the message data based on the unpacked length
                    data = b""
                    while len(data) < length:
                        packet = conn.recv(length - len(data))
                        if not packet:
                            raise ConnectionError("Client disconnected before sending all data.")
                        data += packet

                    # 4. Decode the message and put it in the queue for the GUI
                    if data:
                        message = data.decode('utf-8', errors='ignore')
                        self.message_queue.put(message)

        except (ConnectionResetError, ConnectionError) as e:
            print(f"[-] Client {addr} disconnected: {e}")
        except Exception as e:
            print(f"[!] Error with client {addr}: {e}")
        finally:
            print(f"[-] Connection closed for {addr}")
            self.status_var.set("  Listening on {}:{}".format(HOST, PORT))

    def process_queue(self):
        """ Periodically checks the queue for messages and updates the GUI. """
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self.update_text_area(message)
        except queue.Empty:
            pass
        finally:
            if self.is_running:
                self.root.after(100, self.process_queue)

    def update_text_area(self, message):
        """ Appends a new message to the text area. """
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, f"> {message}\n\n")
        self.text_area.config(state='disabled')
        self.text_area.yview(tk.END)  # Auto-scroll

    def on_closing(self):
        """ Handles the window close event. """
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    main_root = tk.Tk()
    app_instance = App(main_root)
    main_root.mainloop()

