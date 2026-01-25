# speaker.py
import socket
import sounddevice as sd
import numpy as np
import torch
import threading
from TTS.api import TTS
import collections
import port_config as pc_

torch.serialization.add_safe_globals([dict, collections.defaultdict])
from TTS.utils import radam
torch.serialization.add_safe_globals([radam.RAdam])

# Choose a powerful multi-speaker English model
MODEL_NAME = "tts_models/en/vctk/vits"

# Load the model (GPU if available)
tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=torch.cuda.is_available())
MALE_SPEAKER = "p243"

HOST = "0.0.0.0"
PORT = pc_.speaker
STATUS_PORT = pc_.speaker_status

# Shared status variable
status_lock = threading.Lock()
speaker_status = "IDLE"


def set_status(new_status: str):
    """Thread-safe status update."""
    global speaker_status
    with status_lock:
        speaker_status = new_status
    print(f"[STATUS] Speaker -> {new_status}")


def status_server():
    """Continuously serve current BUSY/IDLE status to clients (e.g., mic.py)."""
    print(f"📡 Starting Speaker Status Server on port {STATUS_PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, STATUS_PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                with status_lock:
                    current = speaker_status
                conn.sendall(current.encode("utf-8"))


def speak_text(text):
    """Speak given text and manage status reporting."""
    print(f"🎙 Speaking as {MALE_SPEAKER}: {text}")

    # Send BUSY status before starting speech
    set_status("BUSY")

    # Generate and play speech
    wav = tts.tts(text=text, speaker=MALE_SPEAKER)
    sr = tts.synthesizer.output_sample_rate
    silence = np.zeros(int(0.6 * sr))
    wav = np.concatenate([silence, wav])

    sd.play(wav, samplerate=sr)
    sd.wait()

    # After speaking, set IDLE
    set_status("IDLE")
    print("✅ Done speaking.")


def start_server():
    print(f"🚀 Starting GPU TTS server on {HOST}:{PORT} (voice: {MALE_SPEAKER})")

    # Start status publisher server
    threading.Thread(target=status_server, daemon=True).start()

    # Ensure initial state is IDLE
    set_status("IDLE")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("✅ Waiting for client connections...")

        while True:
            conn, addr = server_socket.accept()
            print(f"🔌 Connected by {addr}")
            with conn:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    text = data.decode("utf-8").strip()
                    if not text:
                        continue
                    if text.lower() == "exit":
                        print("Client disconnected.")
                        break
                    speak_text(text)


if __name__ == "__main__":
    start_server()
