# tts_server_gpu.py
import socket
import sounddevice as sd
import numpy as np
import torch
from TTS.api import TTS
import collections

torch.serialization.add_safe_globals([
    dict, 
    collections.defaultdict
])
from TTS.utils import radam

torch.serialization.add_safe_globals([radam.RAdam])

# Choose a powerful multi-speaker English model
MODEL_NAME = "tts_models/en/vctk/vits"

# Load the model on GPU
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=torch.cuda.is_available())

# Pick a male speaker (try others with tts.speakers)
MALE_SPEAKER = "p243"  # good clear male tone

HOST = "0.0.0.0"
PORT = 5005


def speak_text(text):
    print(f"🎙 Speaking as {MALE_SPEAKER}: {text}")
    wav = tts.tts(text=text, speaker=MALE_SPEAKER)
    
    # Get correct sample rate from synthesizer
    sr = tts.synthesizer.output_sample_rate
    
    # Add 0.3s of silence to smooth start
    silence = np.zeros(int(0.3 * sr))
    wav = np.concatenate([silence, wav])
    
    sd.play(wav, samplerate=sr)
    sd.wait()

def start_server():
    print(f"🚀 Starting GPU TTS server on {HOST}:{PORT} (voice: {MALE_SPEAKER})")
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
