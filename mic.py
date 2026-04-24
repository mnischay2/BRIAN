import socket
import time
import pyaudio
import numpy as np
import struct
import collections
import sys
import scripts.configs.port_config as pc_

def main():
    transcriber_host = "127.0.0.1"
    transcriber_port = pc_.transcriber
    speaker_status_host = "127.0.0.1"
    speaker_status_port = pc_.speaker_status

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    SILENCE_SECONDS = 2
    SILENCE_CHUNKS = int(RATE / CHUNK * SILENCE_SECONDS)
    CALIBRATION_SECONDS = 5
    PRE_SPEECH_PADDING_CHUNKS = int(RATE / CHUNK * 0.5)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    silence_threshold = calibrate_microphone(stream, CALIBRATION_SECONDS, CHUNK, RATE)

    # Connect to the transcriber initially
    transcriber_sock = connect_to_transcriber(transcriber_host, transcriber_port)

    try:
        # --- KEY CHANGES START HERE ---
        # The main loop now handles the status check directly, preventing it from blocking.
        while True:
            # 1. Check speaker status first.
            speaker_status = get_speaker_status(speaker_status_host, speaker_status_port)

            # 2. If the speaker is busy, wait and restart the loop.
            #    This is the non-blocking approach.
            if speaker_status == "BUSY":
                print("[*] Speaker is busy, mic is paused...", end="\r", flush=True)
                time.sleep(0.25)  # Short pause to prevent spamming the status server
                continue # Skip the rest of the loop and check status again.

            # 3. If the speaker is IDLE, proceed with recording.
            print("\n[*] Speaker is idle. Listening for speech...")
            audio_data = record_until_silence(stream, silence_threshold, CHUNK, RATE, PRE_SPEECH_PADDING_CHUNKS, SILENCE_CHUNKS)
            
            # 4. Send the recorded audio for transcription.
            transcriber_sock = send_audio_data(transcriber_sock, audio_data, transcriber_host, transcriber_port)
        # --- KEY CHANGES END HERE ---

    except KeyboardInterrupt:
        print("\n[!] Exiting by user request.")
    finally:
        print("[*] Cleaning up resources.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        if transcriber_sock:
            transcriber_sock.close()

def get_speaker_status(host, port):
    """Connects, gets the status once, and closes."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2) # Prevent long waits
            sock.connect((host, port))
            status = sock.recv(1024).decode('utf-8')
            return status
    except (socket.error, socket.timeout):
        # If the speaker server isn't ready, assume it's idle but log the error.
        print(f"[!] Could not connect to speaker status at {host}:{port}. Assuming IDLE.", end="\r", flush=True)
        return "IDLE"

# This function is no longer needed and has been removed.
# def check_speaker_status(...):

def connect_to_transcriber(host, port):
    while True:
        try:
            print(f"[*] Mic attempting to connect to transcriber at {host}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print("[+] Mic connected to transcriber.")
            return sock
        except Exception as e:
            print(f"[!] Connection to transcriber failed: {e}. Retrying in 5s...")
            time.sleep(5)

def calibrate_microphone(stream, seconds, chunk, rate):
    print(f"[*] Calibrating for {seconds} seconds. Please be quiet...")
    # Clear buffer before calibration
    for _ in range(int(rate / chunk * 1)): # Clear 1 second of buffer
        stream.read(chunk, exception_on_overflow=False)

    noise_levels = [np.abs(np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype=np.int16)).mean() for _ in range(int(rate / chunk * seconds))]

    median_noise = np.median(noise_levels)
    # A more robust threshold calculation
    dynamic_threshold = median_noise * 2.5 + 350
    final_threshold = max(dynamic_threshold, 500) # Set a higher floor

    print(f"[+] Calibration complete. Median noise: {median_noise:.2f}, Threshold: {final_threshold:.2f}")
    return final_threshold

def record_until_silence(stream, silence_threshold, chunk, rate, padding, silence_chunks):
    pre_buffer = collections.deque(maxlen=padding)

    # Wait for speech to start
    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        pre_buffer.append(data)
        if np.abs(np.frombuffer(data, dtype=np.int16)).mean() > silence_threshold:
            print("[+] Speech detected. Recording...")
            break

    frames = list(pre_buffer)
    silence_counter = 0
    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        if np.abs(np.frombuffer(data, dtype=np.int16)).mean() < silence_threshold:
            silence_counter += 1
        else:
            silence_counter = 0
        if silence_counter > silence_chunks:
            print("[*] Silence detected. Stopped recording.")
            break
    return b''.join(frames)

def send_audio_data(sock, audio_data, host, port):
    try:
        length = struct.pack('>I', len(audio_data))
        sock.sendall(length)
        sock.sendall(audio_data)
        print(f"[*] Sent {len(audio_data)} bytes of audio data.")
        return sock
    except (socket.error, BrokenPipeError):
        print("[!] Transcriber disconnected. Reconnecting...")
        if sock:
            sock.close()
        return connect_to_transcriber(host, port)

if __name__ == "__main__":
    main()
