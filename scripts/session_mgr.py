import socket
import struct
import json
import os
import time
import threading
import sys
import datetime
import psycopg2
from psycopg2 import Error
import port_config as pc_
import postgres_config as cfg

class SessionManager:
    def __init__(self):
        self.log_dir = "sessions"
        self.timeout = 1200
        self.current_session = None
        self.session_file = None
        self.session_id_str = None
        self.question_counter = 0
        self.last_activity = None
        self.lock = threading.Lock()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def get_db_connection(self):
        return psycopg2.connect(
            dbname=cfg.TARGET_DB_NAME, 
            **cfg.PG_CREDENTIALS
        )

    def start_new_session(self):
        with self.lock:
            if self.current_session is not None:
                self.save_session()

            self.session_id_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.question_counter = 0
            self.session_file = os.path.join(self.log_dir, f"session_{self.session_id_str}.json")
            self.current_session = []
            self.last_activity = time.time()
            print(f"[*] Starting new session: {self.session_id_str}")

    def add_entry(self, entry_data):
        with self.lock:
            if self.current_session is None:
                self.start_new_session()

            self.current_session.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "interaction": entry_data
            })
            self.last_activity = time.time()
            self.question_counter += 1
            
            self.save_session()
            self.log_to_db(entry_data)
            print(f"[+] Added entry {self.question_counter} to session {self.session_id_str}.")

    def log_to_db(self, entry_data):
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            now = datetime.datetime.now()
            
            question_text = entry_data.get('question', entry_data.get('user_input', ''))
            response_text = entry_data.get('response', entry_data.get('answer', ''))
            
            if not question_text and not response_text:
                question_text = "RAW_DATA"
                response_text = json.dumps(entry_data)

            insert_query = """ 
            INSERT INTO sessions (question_number, date, time, session_id, question, response) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                self.question_counter,
                now.date(),
                now.time(),
                self.session_id_str,
                question_text,
                response_text
            ))
            conn.commit()
        except (Exception, Error) as error:
            print(f"[!] Failed to insert into PostgreSQL: {error}")
        finally:
            if conn:
                cursor.close()
                conn.close()

    def save_session(self):
        if self.session_file and self.current_session is not None:
            try:
                with open(self.session_file, 'w') as f:
                    json.dump(self.current_session, f, indent=4)
            except Exception as e:
                print(f"[!] Error saving session file: {e}")

    def check_timeout(self):
        while True:
            time.sleep(60)
            with self.lock:
                if self.current_session is not None and (time.time() - self.last_activity > self.timeout):
                    print(f"[*] Session timed out.")
                    self.save_session()
                    self.current_session = None
                    self.session_file = None
                    self.session_id_str = None
                    self.question_counter = 0

def handle_client(conn, manager):
    try:
        with conn:
            while True:
                length_bytes = conn.recv(4)
                if not length_bytes:
                    break
                length = struct.unpack('>I', length_bytes)[0]
                data = b""
                while len(data) < length:
                    packet = conn.recv(length - len(data))
                    if not packet:
                        break
                    data += packet
                if len(data) < length:
                    break

                try:
                    interaction = json.loads(data.decode('utf-8'))
                    manager.add_entry(interaction)
                except json.JSONDecodeError as e:
                    print(f"[!] Received malformed JSON data: {e}")

    except (ConnectionResetError, BrokenPipeError):
        pass

def main():
    try:
        host = "127.0.0.1"
        port = pc_.session_mgr
    except KeyError as e:
        print(f"[!] Missing configuration: {e}")
        sys.exit(1)

    manager = SessionManager()
    manager.start_new_session()

    threading.Thread(target=manager.check_timeout, daemon=True).start()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    for i in range(10):
        try:
            server_socket.bind((host, port))
            server_socket.listen()
            print(f"[*] Session Manager listening on {host}:{port}")
            break
        except OSError as e:
            if e.errno == 98:
                time.sleep(1)
            else:
                sys.exit(1)
    else:
        server_socket.close()
        sys.exit(1)

    try:
        while True:
            conn, _ = server_socket.accept()
            threading.Thread(target=handle_client, args=(conn, manager), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()