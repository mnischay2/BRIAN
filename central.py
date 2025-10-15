import socket
import time
import struct
import json
import threading
import port_config as pc_

# --- Service Hosts ---
AI_HOST = "127.0.0.1"
AI_PORT = pc_.ai_handler
TTS_HOST = "127.0.0.1"
TTS_PORT = pc_.speaker
SESSION_HOST = "127.0.0.1"
SESSION_PORT = pc_.session_mgr
SPEAKER_STATUS_HOST = "127.0.0.1"
SPEAKER_STATUS_PORT = pc_.speaker_status

# --- This Server's Config ---
HOST = "0.0.0.0"
TRANSCRIPTION_PORT = pc_.central
UI_PORT = pc_.ui_client

WAKE_WORDS = ["hi brian", "hey brian", "ok brian"]

# --- Globals for UI and Status ---
ui_clients = []
ui_clients_lock = threading.Lock()
current_status = "Idle"
status_lock = threading.Lock()

# -------------------------------
# UI Communication
# -------------------------------
def send_to_all_ui(message):
    """Sends a JSON message to all connected UI clients."""
    with ui_clients_lock:
        payload = json.dumps(message).encode('utf-8')
        length = struct.pack('>I', len(payload))
        dead_clients = []
        for client in ui_clients:
            try:
                client.sendall(length + payload)
            except (BrokenPipeError, ConnectionResetError):
                dead_clients.append(client)
        
        # Remove dead clients
        for client in dead_clients:
            ui_clients.remove(client)

def set_status(new_status, color="grey"):
    """Thread-safe status update and broadcast to UI."""
    global current_status
    with status_lock:
        if current_status == new_status:
            return
        current_status = new_status
        print(f"[Status] -> {new_status}")
    
    send_to_all_ui({
        "type": "status",
        "payload": {"status": new_status, "color": color}
    })

def handle_ui_client(conn, addr):
    """Manages a single UI client connection."""
    print(f"[+] UI client connected from {addr}")
    with ui_clients_lock:
        ui_clients.append(conn)

    try:
        # Keep connection alive
        while True:
            # You can implement ping/pong or receive commands from UI here
            data = conn.recv(1024)
            if not data:
                break
    except (ConnectionResetError, BrokenPipeError):
        print(f"[-] UI client {addr} disconnected.")
    finally:
        with ui_clients_lock:
            if conn in ui_clients:
                ui_clients.remove(conn)

def start_ui_server():
    """Starts the server to listen for UI client connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, UI_PORT))
        s.listen()
        print(f"[*] UI server listening on {HOST}:{UI_PORT}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_ui_client, args=(conn, addr), daemon=True).start()

# -------------------------------
# Speaker Status Monitoring
# -------------------------------
def speaker_status_monitor():
    """Periodically checks speaker status and updates global state."""
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((SPEAKER_STATUS_HOST, SPEAKER_STATUS_PORT))
                status = s.recv(1024).decode('utf-8')
                if status == "BUSY":
                    set_status("Speaking", "#98fb98") # Pale Green
        except (socket.timeout, ConnectionRefusedError):
            # If speaker is not busy or server is down, we aren't "speaking"
            with status_lock:
                if current_status == "Speaking":
                    set_status("Idle", "grey")
        except Exception as e:
            print(f"[Speaker Monitor Error] {e}")
            with status_lock:
                if current_status == "Speaking":
                    set_status("Idle", "grey")
        time.sleep(0.2)

# -------------------------------
# Session Manager Integration
# -------------------------------
def send_to_session_mgr(entry):
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
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tts_sock:
            tts_sock.connect((TTS_HOST, TTS_PORT))
            tts_sock.sendall(text.encode("utf-8"))
    except Exception as e:
        print(f"[TTS Error] {e}")

def send_to_ai(prompt):
    """Sends prompt to AI, streams response to TTS, and returns the full response."""
    full_response = ""
    try:
        set_status("Thinking...", "#ffdab9") # Peach Puff
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ai_sock:
            ai_sock.connect((AI_HOST, AI_PORT))
            ai_sock.sendall(prompt.encode("utf-8"))

            buffer = ""
            while True:
                data = ai_sock.recv(1024)
                if not data: break
                
                chunk = data.decode("utf-8")
                if "\n--- done ---\n" in chunk:
                    chunk = chunk.replace("\n--- done ---\n", "")
                
                print(chunk, end="", flush=True)
                buffer += chunk
                full_response += chunk

                if "." in buffer or "\n" in buffer or "?" in buffer or "!" in buffer:
                    send_to_tts(buffer.strip())
                    buffer = ""
        
        if buffer.strip():
            send_to_tts(buffer.strip())
        
        print("\n--- end of AI response ---")

    except Exception as e:
        print(f"[AI Handler Error] {e}")
    finally:
        # Send final response to the UI
        send_to_all_ui({"type": "chat", "payload": {"role": "assistant", "content": full_response.strip()}})
    
    return full_response.strip()

# -------------------------------
# Transcriber Handler
# -------------------------------
def handle_transcriber(conn, addr):
    print(f"[+] Transcriber connected from {addr}")
    with conn:
        while True:
            length_bytes = conn.recv(4)
            if not length_bytes: break
            length = struct.unpack('>I', length_bytes)[0]
            data = conn.recv(length)
            if not data: break
            
            message = json.loads(data.decode('utf-8'))
            msg_type = message.get("type")
            
            if msg_type == "status":
                set_status(message.get("payload"), "#6495ed") # Cornflower blue
                continue

            elif msg_type == "transcript":
                text = message.get("payload", "").strip()
                if not text: continue

                print(f"🎤 Heard: {text}")
                # Send user's message to UI immediately for responsiveness
                send_to_all_ui({"type": "chat", "payload": {"role": "user", "content": text}})

                lower_text = text.lower()
                matched_wake = next((w for w in WAKE_WORDS if lower_text.startswith(w)), None)
                if matched_wake:
                    trimmed = lower_text[len(matched_wake):].strip()
                    if not trimmed:
                        set_status("Idle", "grey")
                        continue
                    
                    set_status("Awakened", "yellow")
                    # Get the AI's response
                    ai_response = send_to_ai(trimmed)

                    # Now, store the complete interaction as a pair
                    send_to_session_mgr({
                        "question": text,
                        "answer": ai_response
                    })
                else:
                    set_status("Idle", "grey")

    print(f"[-] Transcriber {addr} disconnected.")


# -------------------------------
# Main Server
# -------------------------------
def main():
    threading.Thread(target=start_ui_server, daemon=True).start()
    threading.Thread(target=speaker_status_monitor, daemon=True).start()
    set_status("Idle", "grey")

    print(f"[*] Central server listening for transcriptions on {HOST}:{TRANSCRIPTION_PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, TRANSCRIPTION_PORT))
        server_sock.listen()
        while True:
            conn, addr = server_sock.accept()
            # Each transcriber gets its own thread, though we expect only one.
            threading.Thread(target=handle_transcriber, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()

