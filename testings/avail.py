import pyttsx3
import sys

def list_english_voices():
    """
    Initializes the pyttsx3 engine, finds all available English voices,
    and prints their details including an index number.
    """
    try:
        print("[*] Initializing TTS engine to find available voices...")
        engine = pyttsx3.init()
    except Exception as e:
        print(f"[!!!] CRITICAL ERROR: Could not initialize the pyttsx3 engine.")
        print(f"    Error details: {e}")
        print("    Please ensure you have run 'sudo apt-get install espeak'.")
        sys.exit(1)

    voices = engine.getProperty('voices')
    print("[+] Found the following English voices on your system:")
    print("-" * 50)
    
    english_voice_found = False
    for index, voice in enumerate(voices):
        # Check if 'en' is in any of the language codes for the voice
        if any('en' in lang for lang in voice.languages):
            english_voice_found = True
            print(f"  INDEX: {index}")
            print(f"    ID: {voice.id}")
            print(f"    Name: {voice.name}")
            print(f"    Languages: {voice.languages}")
            print("-" * 50)

    if not english_voice_found:
        print("[!] No English voices were found on your system.")
        print("    Please ensure the 'espeak' package and its English voice data are installed correctly.")

if __name__ == "__main__":
    list_english_voices()

