import sounddevice as sd
import numpy as np
import soundfile as sf
import torch
import queue
import tempfile
import os
import webrtcvad

from nemo.collections.asr.models import EncDecRNNTBPEModel

# Load NeMo ASR model
print("🔁 Loading ASR model...")
asr_model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_fastconformer_transducer_large")
print("✅ ASR model loaded!")

# Audio stream settings
samplerate = 16000
block_duration = 0.5  # seconds
block_size = int(samplerate * block_duration)
channels = 1
q = queue.Queue()

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # 0: less aggressive, 3: most aggressive

# Audio callback to collect blocks
def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}")
    q.put(indata.copy())

# Frame class for VAD
class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(audio_bytes, sample_rate, frame_duration_ms=30):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n <= len(audio_bytes):
        yield Frame(audio_bytes[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

# Speech/silence detection
def detect_speech(audio, sample_rate):
    if isinstance(audio, np.ndarray) and audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)
    audio_bytes = audio.tobytes()
    frames = list(frame_generator(audio_bytes, sample_rate))
    speech_frames = [f for f in frames if vad.is_speech(f.bytes, sample_rate)]
    return len(speech_frames) > 0

# Transcribe from file
def transcribe_wav(path):
    return asr_model.transcribe([path])[0].text.lower()


# Main logic
wake_word = "hi bodhi"
buffer_seconds = 2
buffer_blocks = int(buffer_seconds / block_duration)
audio_buffer = []

print("🎙️ Say 'Hi Bodhi' to activate...")

with sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback,
                    dtype='float32', blocksize=block_size):
    while True:
        block = q.get()
        audio_buffer.append(block)

        if len(audio_buffer) > buffer_blocks:
            audio_buffer.pop(0)

        buffered_audio = np.concatenate(audio_buffer, axis=0)

        # Skip if silent
        if not detect_speech(buffered_audio, samplerate):
            continue

        # Save temp file for ASR check
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, buffered_audio, samplerate)
            tmp_path = tmpfile.name

        try:
            print("📝 Checking for wake word...")
            text = transcribe_wav(tmp_path)
            os.remove(tmp_path)
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            continue

        print(f"👂 Heard: {text.strip()}")

        if wake_word in text:
            print("🟢 Wake word detected! Listening for command...")
            print("🎤 Speak your command. I'll stop listening after a short silence...")

            # Start recording full command
            command_audio = []
            silence_counter = 0
            max_silence_blocks = 4  # ~3 seconds

            while True:
                command_block = q.get()
                command_audio.append(command_block)

                if not detect_speech(command_block, samplerate):
                    silence_counter += 1
                    print(f"🛑 Silence block ({silence_counter}/{max_silence_blocks})")
                else:
                    print(f"🎙️ Speech detected.")
                    silence_counter = 0

                if silence_counter >= max_silence_blocks:
                    break

            print("📦 Command captured. Transcribing...")

            full_command = np.concatenate(command_audio, axis=0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as command_file:
                sf.write(command_file.name, full_command, samplerate)
                command_path = command_file.name

            try:
                command_text = transcribe_wav(command_path)
                print(f"💬 Command: {command_text.strip()}")
            except Exception as e:
                print(f"❌ Transcription error: {e}")

            os.remove(command_path)
            print("🔁 Listening again. Say 'Hi Bodhi' to activate.")
