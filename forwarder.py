#!/home/nischay/linenv/bin/python3
import socket
import struct

# --- Server Configuration ---
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = 5002       # Port for this service

def handle_transcriber(conn, addr):
    """
    Handles the connection from the transcription server.
    """
    print(f"[+] Transcriber connected from {addr}")
    try:
        while True:
            # Receive the 4-byte length prefix
            length_bytes = conn.recv(4)
            if not length_bytes:
                print("[-] Transcriber disconnected.")
                break
            
            length = struct.unpack('>I', length_bytes)[0]
            
            # Receive the text data
            data = b""
            while len(data) < length:
                packet = conn.recv(length - len(data))
                if not packet:
                    print("[-] Transcriber disconnected before all data was sent.")
                    break
                data += packet
            
            if data:
                text = data.decode('utf-8')
                print(f"TRANSCRIPTION RECEIVED: {text}")

    except ConnectionResetError:
        print(f"[-] Transcriber {addr} connection was reset.")
    except Exception as e:
        print(f"[!] An error occurred with {addr}: {e}")
    finally:
        print(f"[*] Closing connection with {addr}.")
        conn.close()

def main():
    """
    Main loop to start the server and wait for the transcriber.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[*] Forwarder listening on {HOST}:{PORT}")

        while True:
            try:
                conn, addr = s.accept()
                handle_transcriber(conn, addr)
            except KeyboardInterrupt:
                print("\n[!] Shutting down forwarder server.")
                break
            except Exception as e:
                print(f"[!] Server loop error: {e}")

if __name__ == "__main__":
    main()
