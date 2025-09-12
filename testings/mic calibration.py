import pyaudio
import numpy as np
import time

# --- Audio Configuration ---
# These should match the settings in your main mic.py
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def run_tuner():
    """
    Listens to the microphone and prints the current amplitude
    to help you find the right SILENCE_THRESHOLD.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("--- Microphone Amplitude Tuner ---")
    print("This will help you find the best SILENCE_THRESHOLD for your environment.")
    print("Press Ctrl+C to exit.")
    print("\nStep 1: Be quiet and observe the 'silent' amplitude...")
    time.sleep(3) # Give user time to read

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate the average amplitude for the current chunk
            amplitude = np.abs(audio_data).mean()
            
            # Print the amplitude. The formatting helps create a simple bar graph.
            bar = '#' * int(amplitude / 100)
            print(f"Current Amplitude: {amplitude:<5.0f} | {bar}")
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n--- Tuner Stopped ---")
        print("Step 2: Now, speak normally and see how high the numbers get.")
        print("Your ideal SILENCE_THRESHOLD should be higher than your silent average,")
        print("but much lower than your speaking average.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    run_tuner()
