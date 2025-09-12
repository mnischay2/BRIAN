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
SILENCE_SECONDS = 2  # How many seconds of silence triggers the end of a phrase
SILENCE_CHUNKS = int(RATE / CHUNK * SILENCE_SECONDS)
CALIBRATION_SECONDS = 10 # How long to listen to calibrate the noise floor
PRE_SPEECH_PADDING_CHUNKS = int(RATE / CHUNK * 0.5) # 0.5 seconds of padding

# --- Network Configuration ---
TRANSCRIBER_HOST = "127.0.0.1"
TRANSCRIBER_PORT = 5001

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

def calibrate_microphone(p, stream):
    """Listens for a few seconds to determine the ambient noise level."""
    print(f"[*] Calibrating for {CALIBRATION_SECONDS} seconds. Please be quiet...")
    
    noise_levels = []
    for _ in range(0, int(RATE / CHUNK * CALIBRATION_SECONDS)):
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        noise_levels.append(np.abs(audio_data).mean())
    
    # Calculate the average noise level and add a buffer
    ambient_noise = np.mean(noise_levels)
    
    # The threshold is the ambient noise plus a margin, or a minimum value.
    # The multiplier (e.g., 2.5) makes it less sensitive to small noise spikes.
    # The adder (e.g., 200) provides a fixed floor.
    dynamic_threshold = ambient_noise * 2.5 + 200
    
    # Ensure the threshold is at least a reasonable minimum
    final_threshold = max(dynamic_threshold, 400) 

    print(f"[+] Calibration complete. Ambient noise level: {ambient_noise:.2f}")
    print(f"[+] Dynamic silence threshold set to: {final_threshold:.2f}")
    
    return final_threshold

def record_until_silence(stream, silence_threshold):
    """
    Waits for speech to start, records it, and stops when silence is detected.
    This prevents sending silent audio chunks.
    """
    print("[*] Waiting for speech...")
    
    # Use a deque to store a small buffer of audio before speech starts.
    # This prevents the first sound from being cut off.
    pre_buffer = collections.deque(maxlen=PRE_SPEECH_PADDING_CHUNKS)
    
    # --- 1. Wait for speech to start (Voice Activity Detection) ---
    while True:
        data = stream.read(CHUNK)
        pre_buffer.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()
        
        # If amplitude is above threshold, speech has started.
        if amplitude > silence_threshold:
            print("[+] Speech detected. Recording...")
            break
            
    # --- 2. Record the speech until silence is detected again ---
    frames = list(pre_buffer)
    silence_counter = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Check amplitude to detect silence
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()

        if amplitude < silence_threshold:
            silence_counter += 1
        else:
            # Reset counter if speech is detected
            silence_counter = 0

        if silence_counter > SILENCE_CHUNKS:
            print(f"[*] Silence detected. Stopped recording.")
            break

    return b''.join(frames)

def send_audio_data(sock, audio_data):
    """Sends the raw audio data to the server with a length prefix."""
    try:
        # Pack the length of the data into a 4-byte integer
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
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    # Calibrate once at the start
    silence_threshold = calibrate_microphone(p, stream)
    
    sock = connect_to_transcriber()

    try:
        while True:
            # Record audio based on the dynamic threshold
            audio_data = record_until_silence(stream, silence_threshold)
            
            # Send the recorded data
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

