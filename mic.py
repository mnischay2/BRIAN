#mic.py
import socket
import time
import pyaudio
import numpy as np
import struct
import collections
import yaml
import sys
import port_config as pc_

def main():
    

    mic_config = pc_.mic
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

    sock = connect_to_transcriber(transcriber_host, transcriber_port)
    status_sock = connect_to_speaker_status(speaker_status_host, speaker_status_port)

    try:
        while True:
            status_sock = check_speaker_status(status_sock, speaker_status_host, speaker_status_port)
            audio_data = record_until_silence(stream, silence_threshold, CHUNK, RATE, PRE_SPEECH_PADDING_CHUNKS, SILENCE_CHUNKS)
            sock = send_audio_data(sock, audio_data, transcriber_host, transcriber_port)
    except KeyboardInterrupt:
        print("\n[!] Exiting by user request.")
    finally:
        print("[*] Cleaning up resources.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        if status_sock:
            status_sock.close()

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

def connect_to_speaker_status(host, port):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            return sock
        except Exception as e:
            print(f"[!] Could not connect to speaker status server: {e}. Retrying...")
            time.sleep(3)

def check_speaker_status(sock, host, port):
    while True:
        try:
            if sock:
                sock.close()
            sock = connect_to_speaker_status(host, port)
            status = sock.recv(1024).decode('utf-8')
            if status == "BUSY":
                print("[*] Speaker is busy, waiting...")
                time.sleep(0.5)
                continue
            else:
                return sock
        except (socket.error, BrokenPipeError):
            print("[!] Speaker status connection lost. Reconnecting...")
            time.sleep(1)

def calibrate_microphone(stream, seconds, chunk, rate):
    print(f"[*] Calibrating for {seconds} seconds. Please be quiet...")
    for _ in range(5):
        stream.read(chunk, exception_on_overflow=False)

    noise_levels = [np.abs(np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype=np.int16)).mean() for _ in range(int(rate / chunk * seconds))]

    median_noise = np.median(noise_levels)
    dynamic_threshold = median_noise * 2.0 + 300
    final_threshold = max(dynamic_threshold, 400)

    print(f"[+] Calibration complete. Median noise: {median_noise:.2f}, Threshold: {final_threshold:.2f}")
    return final_threshold

def record_until_silence(stream, silence_threshold, chunk, rate, padding, silence_chunks):
    print("[*] Waiting for speech...")
    pre_buffer = collections.deque(maxlen=padding)

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
        sock.close()
        return connect_to_transcriber(host, port)

if __name__ == "__main__":
    main()
