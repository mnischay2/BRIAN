#!/home/nischay/linenv311/bin/python
import socket
import time
import pyaudio
import numpy as np
import struct
import collections

# --- Audio Configuration ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_SECONDS = 2
SILENCE_CHUNKS = int(RATE / CHUNK * SILENCE_SECONDS)
CALIBRATION_SECONDS = 5
PRE_SPEECH_PADDING_CHUNKS = int(RATE / CHUNK * 0.5)
WARMUP_CHUNKS = 10

# --- Network Configuration ---
TRANSCRIBER_HOST = "127.0.0.1"
TRANSCRIBER_PORT = 5001
SPEAKER_STATUS_HOST = "127.0.0.1"
SPEAKER_STATUS_PORT = 5004

def is_speaker_active():
    """Checks if the speaker service is currently outputting audio."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5) # Don't wait forever
            s.connect((SPEAKER_STATUS_HOST, SPEAKER_STATUS_PORT))
            status = s.recv(1).decode('utf-8')
            return status == "1"
    except (socket.timeout, ConnectionRefusedError, ConnectionResetError):
        # If we can't connect, assume it's not speaking.
        return False
    except Exception as e:
        print(f"[!] Speaker status check error: {e}")
        return False

def connect_to_transcriber():
    """Attempts to connect to the transcription server with retries."""
    while True:
        try:
            print(f"[*] Attempting to connect to transcriber at {TRANSCRIBER_HOST}:{TRANSCRIBER_PORT}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((TRANSCRIBER_HOST, TRANSCRIBER_PORT))
            print("[+] Connected to transcriber.")
            return sock
        except ConnectionRefusedError:
            print("[!] Connection refused. Is transcribe.py running? Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"[!] Connection failed: {e}. Retrying in 5s...")
            time.sleep(5)

def calibrate_microphone(stream):
    """Calculates the ambient noise level using the median to resist spikes."""
    print(f"[*] Calibrating for {CALIBRATION_SECONDS} seconds. Please be quiet...")
    
    # Warm-up phase to discard initial mic pops
    for _ in range(WARMUP_CHUNKS):
        stream.read(CHUNK, exception_on_overflow=False)

    noise_levels = []
    for _ in range(int(RATE / CHUNK * CALIBRATION_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        noise_levels.append(np.abs(audio_data).mean())

    median_noise = np.median(noise_levels)
    # A more robust formula based on the stable median
    dynamic_threshold = median_noise * 2.0 + 300 
    final_threshold = max(dynamic_threshold, 400) # Ensure a minimum floor

    print(f"[+] Calibration complete. Median noise level: {median_noise:.2f}")
    print(f"[+] Dynamic silence threshold set to: {final_threshold:.2f}")
    return final_threshold

def record_until_silence(stream, silence_threshold):
    """
    Waits for speech to start, records it, and stops when silence is detected.
    """
    pre_buffer = collections.deque(maxlen=PRE_SPEECH_PADDING_CHUNKS)
    
    # --- 1. Wait for speaker to finish ---
    while is_speaker_active():
        print("[/] Speaker is active. Waiting...")
        time.sleep(0.5)

    print("[*] Waiting for speech...")
    # --- 2. Wait for speech to start (Voice Activity Detection) ---
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        pre_buffer.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() > silence_threshold:
            print("[+] Speech detected. Recording...")
            break
            
    # --- 3. Record the speech until silence is detected again ---
    frames = list(pre_buffer)
    silence_counter = 0
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0
        if silence_counter > SILENCE_CHUNKS:
            print("[*] Silence detected. Stopped recording.")
            break
    return b''.join(frames)

def send_audio_data(sock, audio_data):
    """Sends the raw audio data to the server with a length prefix."""
    try:
        length = struct.pack('>I', len(audio_data))
        sock.sendall(length)
        sock.sendall(audio_data)
        print(f"[*] Sent {len(audio_data)} bytes of audio data.")
        return sock
    except (socket.error, BrokenPipeError, ConnectionResetError) as e:
        print(f"[!] Transcriber disconnected: {e}. Reconnecting...")
        sock.close()
        return connect_to_transcriber()

def main():
    """Main loop to record, connect, and send audio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    silence_threshold = calibrate_microphone(stream)
    sock = connect_to_transcriber()

    try:
        while True:
            audio_data = record_until_silence(stream, silence_threshold)
            sock = send_audio_data(sock, audio_data)
    except KeyboardInterrupt:
        print("\n[!] Exiting by user request.")
    finally:
        print("[*] Cleaning up resources.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()

if __name__ == "__main__":
    main()

