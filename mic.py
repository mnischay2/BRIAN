
import socket
import pyaudio

# Config
DEVICE_A_IP = "localhost"  
PORT_SPEAKER = 7001


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

# Send mic audio to Device A
def send_to_speaker():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((DEVICE_A_IP, PORT_SPEAKER))
    print(f"📤 Sending mic audio to {DEVICE_A_IP}:{PORT_SPEAKER}")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    try:
        while True:
            data = stream.read(CHUNK)
            s.sendall(data)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        s.close()


if __name__ == "__main__":
    send_to_speaker()