import socket
import torch
import numpy as np
import soundfile as sf
import io
import time
import queue
import threading
from nemo.collections.asr.models import EncDecRNNTBPEModel

# CONFIG
DEVICE_B_IP = "localhost"  # Replace with Device B's IP
PORT_MIC = 7001
PORT_RESPONSE = 9000

CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3  # Buffer 3 seconds of audio
BUFFER_SIZE = SAMPLE_RATE * 2 * BUFFER_SECONDS  # 2 bytes per sample

WAKE_WORDS = ["hi BRIAN", "hey BRIAN", "brian", "brain"]

# Load NeMo ASR model
print("⏳ Loading ASR model...")
asr_model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_fastconformer_transducer_large")
print("✅ ASR model loaded!")

audio_buffer = bytearray()
data_queue = queue.Queue()

# Receive audio MIC.PY
def receive_audio_stream():
    global audio_buffer

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", PORT_MIC))
    server.listen(1)
    print(f"🎧 Waiting for mic stream on port {PORT_MIC}...")
    conn, addr = server.accept()
    print(f"✅ Mic stream connected from {addr}")

    while True:
        data = conn.recv(CHUNK_SIZE)
        if not data:
            break
        audio_buffer.extend(data)

        if len(audio_buffer) >= BUFFER_SIZE:
            chunk = bytes(audio_buffer[:BUFFER_SIZE])
            del audio_buffer[:BUFFER_SIZE]
            data_queue.put(chunk)

# Transcribe audio chunk to text
def transcribe(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    with torch.no_grad():
        transcript = asr_model.transcribe([audio_np])[0]
    return transcript.text.lower().strip()

# Send response text to FORWARDER.PY
def send_response(text):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((DEVICE_B_IP, PORT_RESPONSE))
        s.sendall(text.encode("utf-8"))
        s.close()
    except Exception as e:
        print(f"❌ Could not send response to Device B: {e}")

# Compute Root Mean Square (RMS) energy to detect silence
def rms_energy(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(audio_np ** 2))

# Main processing loop
def process_audio():
    listening = False
    command_audio = bytearray()
    silence_start = None
    SILENCE_THRESHOLD = 500  # Tune this threshold as needed
    SILENCE_DURATION = 3.0   # Stop listening after 3 seconds of silence

    while True:
        if not data_queue.empty():
            audio_chunk = data_queue.get()

            if not listening:
                print("📝 Checking for wake word...")
                transcript = transcribe(audio_chunk)
                print(f"👂 Heard: {transcript}")
                matched_wake_word = next((word for word in WAKE_WORDS if word in transcript), None)

                if matched_wake_word:
                    print(f"🟢 Wake word '{matched_wake_word}' detected! Listening for command...")
                    listening = True
                    command_audio = bytearray()
                    silence_start = None
                    send_response("Hi")
                continue

            # Accumulate command audio
            command_audio.extend(audio_chunk)

            # Check for silence
            energy = rms_energy(audio_chunk)
            print(f"🔊 Energy: {energy:.2f}")

            if energy < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    print("⏹️ Silence detected. Processing command...")
                    command_text = transcribe(bytes(command_audio)).strip().lower()
                    for word in WAKE_WORDS:
                        command_text = command_text.replace(word, "")
                    command_text = command_text.strip()
                    print(f"🗣️ Command: {command_text}")
                    send_response(command_text if command_text else "No command detected.")
                    listening = False
            else:
                silence_start = None  # Reset timer if speech detected

# Start everything
if __name__ == "__main__":
    threading.Thread(target=receive_audio_stream, daemon=True).start()
    process_audio()