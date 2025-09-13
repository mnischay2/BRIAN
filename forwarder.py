#!/home/nischay/linenv311/bin/python
import socket
import struct
import tkinter as tk
from tkinter import scrolledtext, font as tkfont
import threading
import queue
import requests
import json
import time
import re

# --- Configuration ---
TRANSCRIBER_LISTENER_HOST = "0.0.0.0"
TRANSCRIBER_LISTENER_PORT = 5002

SPEAKER_HOST = "127.0.0.1"
SPEAKER_PORT = 5003

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
WAKE_WORDS = ["Brian", "brian"] # Case-sensitive wake words

# --- Color and Font Scheme ---
BG_COLOR = "#1e1e1e"
TEXT_AREA_BG = "#2a2a2a"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#00aaff"
FONT_FAMILY = "Roboto"

class VoiceAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("B.R.I.A.N. - Bio-Responsive Intelligent Assistant Network")
        self.root.geometry("800x600")
        self.root.configure(bg=BG_COLOR)

        # --- Font Configuration ---
        self.default_font = tkfont.Font(family=FONT_FAMILY, size=12)
        self.status_font = tkfont.Font(family=FONT_FAMILY, size=10, weight="bold")
        self.title_font = tkfont.Font(family=FONT_FAMILY, size=16, weight="bold")

        # --- Main UI Elements ---
        title_label = tk.Label(root, text="B.R.I.A.N.", font=self.title_font, bg=BG_COLOR, fg=ACCENT_COLOR)
        title_label.pack(pady=(10, 5))

        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=self.default_font,
                                                 bg=TEXT_AREA_BG, fg=TEXT_COLOR, insertbackground='white',
                                                 bd=0, relief=tk.FLAT, padx=10, pady=10)
        self.text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # --- Status Indicators Frame ---
        status_frame = tk.Frame(root, bg=BG_COLOR)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.wake_status_label = tk.Label(status_frame, text="WAKE: SLEEPING", font=self.status_font, bg=BG_COLOR)
        self.wake_status_label.pack(side=tk.LEFT, padx=(5, 20))

        self.llm_status_label = tk.Label(status_frame, text="LLM: IDLE", font=self.status_font, bg=BG_COLOR)
        self.llm_status_label.pack(side=tk.LEFT)

        # --- State and Networking ---
        self.message_queue = queue.Queue()
        self.speaker_sock = None
        self.is_awake = False

        threading.Thread(target=self.connect_to_speaker_service, daemon=True).start()
        threading.Thread(target=self.start_transcriber_listener, daemon=True).start()
        
        self.root.after(100, self.process_message_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_wake_status("SLEEPING", "#e07b7b") # Red
        self.update_llm_status("IDLE", "#a0a0a0") # Grey

    def update_wake_status(self, text, color):
        self.wake_status_label.config(text=f"WAKE: {text}", fg=color)

    def update_llm_status(self, text, color):
        self.llm_status_label.config(text=f"LLM: {text}", fg=color)

    def update_text_area(self, user, message):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, f"{user}: {message}\n\n")
        self.text_area.config(state='disabled')
        self.text_area.yview(tk.END)

    def process_message_queue(self):
        try:
            user, message = self.message_queue.get_nowait()
            self.update_text_area(user, message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_message_queue)

    def connect_to_speaker_service(self):
        while True:
            try:
                self.update_llm_status(f"CONNECTING TO SPEAKER", "#e0d37b")
                self.speaker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.speaker_sock.connect((SPEAKER_HOST, SPEAKER_PORT))
                self.update_llm_status("IDLE", "#a0a0a0")
                break 
            except Exception:
                time.sleep(3)
    
    def send_to_speaker(self, text):
        if not self.speaker_sock:
            self.message_queue.put(("System", "Error: Not connected to speaker service."))
            self.connect_to_speaker_service()
            return
        try:
            encoded_text = text.encode('utf-8')
            length_prefix = struct.pack('>I', len(encoded_text))
            self.speaker_sock.sendall(length_prefix)
            self.speaker_sock.sendall(encoded_text)
        except (socket.error, BrokenPipeError) as e:
            self.message_queue.put(("System", f"Speaker disconnected: {e}. Reconnecting..."))
            self.speaker_sock.close()
            self.speaker_sock = None
            self.connect_to_speaker_service()

    def clean_text_for_speech(self, text):
        """
        Removes symbols that are poorly handled by TTS, while keeping
        essential punctuation for natural speech flow.
        """
        # This regex removes any character that is NOT a letter, number, whitespace,
        # or one of the essential punctuation marks (.,?!').
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s.,?!']", " ", text)
        # Replace multiple spaces that might result from the substitution with a single space.
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def llm_worker(self, command_text):
        try:
            self.root.after(0, self.update_llm_status, "THINKING...", "#e0d37b") # Yellow
            payload = { "model": OLLAMA_MODEL, "prompt": command_text, "stream": False }
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
            response.raise_for_status()
            llm_response_text = json.loads(response.text).get("response", "Sorry, I had trouble thinking of a response.").strip()
            
            # Clean the text to be spoken, but keep the original for the GUI
            speech_text = self.clean_text_for_speech(llm_response_text)
            
            self.message_queue.put(("Assistant", llm_response_text))
            self.root.after(0, self.update_llm_status, "SPEAKING", "#7bcee0") # Cyan
            self.send_to_speaker(speech_text)

        except requests.exceptions.RequestException as e:
            self.message_queue.put(("System", f"Error connecting to LLM: {e}"))
        finally:
            self.is_awake = False
            self.root.after(1000, self.update_wake_status, "SLEEPING", "#e07b7b") # Red
            self.root.after(1000, self.update_llm_status, "IDLE", "#a0a0a0") # Grey

    def process_transcription(self, transcribed_text):
        if self.is_awake:
            self.message_queue.put(("You", transcribed_text))
            threading.Thread(target=self.llm_worker, args=(transcribed_text,)).start()
        else:
            for wake_word in WAKE_WORDS:
                if wake_word in transcribed_text:
                    self.is_awake = True
                    self.message_queue.put(("System", f"Wake word '{wake_word}' detected. Listening..."))
                    self.update_wake_status("LISTENING", "#7be08a") # Green
                    return

    def handle_transcriber_client(self, conn, addr):
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
                        text = data.decode('utf-8').strip()
                        if text:
                            self.process_transcription(text)
        except ConnectionResetError:
            pass # Client disconnected, handled by the main loop
        finally:
            pass # Main loop will listen for a new connection
            
    def start_transcriber_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((TRANSCRIBER_LISTENER_HOST, TRANSCRIBER_LISTENER_PORT))
            s.listen()
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self.handle_transcriber_client, args=(conn, addr), daemon=True).start()

    def on_closing(self):
        if self.speaker_sock:
            self.speaker_sock.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistantGUI(root)
    root.mainloop()

