#central.py
import socket
import time
import struct
import json
import port_config as pc_

AI_HOST = "127.0.0.1"
AI_PORT = pc_.ai_handler

TTS_HOST = "127.0.0.1"
TTS_PORT = pc_.speaker

SESSION_HOST = "127.0.0.1"
SESSION_PORT = pc_.session_mgr

HOST = "0.0.0.0"
PORT = pc_.central

WAKE_WORDS = ["hi brian", "hey brian", "ok brian"]


# -------------------------------
# Session Manager Integration
# -------------------------------
def send_to_session_mgr(entry):
    """Send an interaction entry to session manager via socket."""
    try:
        payload = json.dumps(entry).encode("utf-8")
        length = struct.pack(">I", len(payload))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SESSION_HOST, SESSION_PORT))
            s.sendall(length + payload)
    except ConnectionRefusedError:
        print("[⚠️] Session manager not running or unreachable.")
    except Exception as e:
        print(f"[SessionMgr Error] {e}")


# -------------------------------
# Core Communication Functions
# -------------------------------
def send_to_tts(text):
    """Send text to the TTS server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tts_sock:
            tts_sock.connect((TTS_HOST, TTS_PORT))
            tts_sock.sendall(text.encode("utf-8"))
    except Exception as e:
        print(f"[TTS Error] {e}")


def send_to_ai(prompt):
    """Send prompt to AI handler and stream response to TTS and SessionMgr."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ai_sock:
            ai_sock.connect((AI_HOST, AI_PORT))
            ai_sock.sendall(prompt.encode("utf-8"))

            print(f"🧠 AI ({prompt[:50]}...): ", end="", flush=True)
            buffer = ""
            full_response = ""

            while True:
                data = ai_sock.recv(1024)
                if not data:
                    break

                chunk = data.decode("utf-8")

                # Stop at the termination marker
                if "\n--- done ---\n" in chunk:
                    chunk = chunk.replace("\n--- done ---\n", "")
                    if chunk.strip():
                        buffer += chunk
                    if buffer.strip():
                        send_to_tts(buffer.strip())
                        full_response += buffer.strip() + " "
                    buffer = ""
                    break

                print(chunk, end="", flush=True)
                buffer += chunk

                # Send periodically to TTS
                if "." in chunk or "\n" in chunk or len(buffer) > 150:
                    text_to_send = buffer.strip()
                    if text_to_send:
                        send_to_tts(text_to_send)
                        full_response += text_to_send + " "
                    buffer = ""
                    time.sleep(0.1)

            if buffer.strip():
                send_to_tts(buffer.strip())
                full_response += buffer.strip()

            # Log assistant response to session manager
            send_to_session_mgr({
                "role": "assistant",
                "content": full_response.strip()
            })

            print("\n--- end of AI response ---")

    except Exception as e:
        print(f"[AI Handler Error] {e}")


# -------------------------------
# Transcriber Handler
# -------------------------------
def handle_transcriber(conn, addr):
    """Receive transcribed text and process wake word."""
    print(f"[+] Transcriber connected from {addr}")
    with conn:
        while True:
            # Each message is prefixed with 4-byte length
            length_bytes = conn.recv(4)
            if not length_bytes:
                print("[-] Transcriber disconnected.")
                break

            length = struct.unpack('>I', length_bytes)[0]
            data = conn.recv(length)
            if not data:
                break

            text = data.decode('utf-8').strip()
            print(f"🎤 Heard: {text}")

            # Log user input to session manager
            send_to_session_mgr({
                "role": "user",
                "content": text
            })

            # Check for wake word
            lower_text = text.lower()
            matched_wake = next((w for w in WAKE_WORDS if lower_text.startswith(w)), None)
            if matched_wake:
                trimmed = lower_text[len(matched_wake):].strip()
                if not trimmed:
                    print("[~] Wake word detected but no prompt. Ignoring.")
                    continue
                print(f"✅ Wake word detected ('{matched_wake}'). Sending to AI: {trimmed}")
                send_to_ai(trimmed)
            else:
                print("[~] No wake word detected. Ignoring input.")


# -------------------------------
# Main Server
# -------------------------------
def main():
    print(f"[*] Central server listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        while True:
            conn, addr = server_sock.accept()
            handle_transcriber(conn, addr)


if __name__ == "__main__":
    main()
