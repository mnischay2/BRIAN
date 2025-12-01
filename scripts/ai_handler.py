import socket
import threading
import requests
import json
import port_config as pc_ 
import os
import psycopg2
import postgres_config as cfg 
from dotenv import load_dotenv

# --- RAG Integration ---
from rag_query import hybrid_search 
# ---------------------

load_dotenv()

HOST = "127.0.0.1"
PORT = pc_.ai_handler_port 

# Get Ollama config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "llama3")

def get_recent_history(limit=5):
    """
    Fetches the last few interactions from the 'sessions' table 
    to provide conversation history/memory to the LLM.
    """
    conn = None
    history_text = ""
    try:
        conn = psycopg2.connect(dbname=cfg.TARGET_DB_NAME, **cfg.PG_CREDENTIALS)
        cur = conn.cursor()
        
        # Fetch last N rows
        query = """
            SELECT question, response 
            FROM sessions 
            ORDER BY id DESC 
            LIMIT %s
        """
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        
        # Reverse to chronological order (oldest -> newest)
        rows.reverse()
        
        if rows:
            formatted_lines = []
            for q, a in rows:
                q_text = q if q else "[No Question]"
                a_text = a if a else "[No Response]"
                formatted_lines.append(f"User: {q_text}\nAssistant: {a_text}")
            history_text = "\n\n".join(formatted_lines)
        else:
            history_text = "No previous conversation history."
            
        cur.close()
    except Exception as e:
        print(f"  ! DB History Error: {e}")
        history_text = "[Error retrieving history]"
    finally:
        if conn: conn.close()
        
    return history_text

def handle_client(conn, addr):
    print(f"[+] Connected by {addr}")
    try:
        # 1. Receive user message
        prompt = conn.recv(4096).decode().strip()
        if not prompt:
            conn.close()
            return

        print(f"\n[User Prompt] {prompt}")

        # 2. --- HISTORY STEP: Get Recent Chat ---
        print("  > Fetching Conversation History...")
        history_block = get_recent_history(limit=5)

        # 3. --- RAG STEP: Get Context from Knowledge Base ---
        print("  > Searching Knowledge Base (Resumes, Books, PDFs)...")
        try:
            # Fetch top 3 most relevant chunks
            context_chunks = hybrid_search(prompt, limit=5)
            
            if context_chunks:
                context_block = "\n\n---\n\n".join(context_chunks)
                print(f"  > Found {len(context_chunks)} relevant documents/sections.")
            else:
                context_block = "No specific documents found in the database matching this query."
                print("  > No context found.")
        except Exception as e:
            print(f"  ! RAG Error: {e}")
            context_block = "Error retrieving documents."

        # 4. --- PROMPT ENGINEERING ---
        # Optimized for generic document handling (Resumes, Books, etc.)
        final_system_msg = """
        You are BODHI, an intelligent Knowledge Assistant.
        
        You have access to two sources of information:
        1. "Recent Conversation History" (Short-term memory of this chat).
        2. "Retrieved Knowledge" (Excerpts from uploaded files like Resumes, Books, PDFs, etc.).

        INSTRUCTIONS:
        - Analyze the "Retrieved Knowledge" carefully. It may contain candidate details, technical documentation, or narrative text.
        - Answer the User's Question based *primarily* on the "Retrieved Knowledge".
        - If the user asks about a specific person (e.g., from a resume), assume the "Retrieved Knowledge" contains the correct data. but proceed to check in recent conversation history in case there is nothing in retrieved knowledge.
        - If the answer is NOT in the retrieved documents, explicitly state: "I could not find that information in the uploaded documents," and then provide a general answer based on your own training if applicable.
        - Keep responses professional, concise, and direct. Do not use markdown (bold/italic) or asterisks.
        """.strip()
        
        final_user_msg = f"""
        Recent Conversation History:
        {history_block}

        Retrieved Knowledge:
        {context_block}

        ---
        User Question: 
        {prompt}
        """

        payload = {
            "model": LLM_MODEL_ID,
            "messages": [
                {"role": "system", "content": final_system_msg},
                {"role": "user", "content": final_user_msg}
            ],
            "stream": True
        }

        # 5. Stream Response
        print(f"  > Streaming response from {LLM_MODEL_ID}...")
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            conn.sendall(token.encode("utf-8"))
                        
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue 

        # Send a final "done" marker
        conn.sendall(b"\n--- done ---\n")
        
    except (socket.error, ConnectionResetError) as e:
        print(f"[!] Client connection error: {e}")
    except Exception as e:
        print(f"[!] Unhandled Error: {e}")
    finally:
        conn.close()
        print(f"[-] Disconnected {addr}")

def start_server():
    print(f"[*] Starting Bodhi AI Server on {HOST}:{PORT}")
    print(f"[*] RAG System: Active (Knowledge Base)")
    print(f"[*] Memory System: Active (Session History)")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, PORT))
        except OSError as e:
            print(f"❌ FATAL ERROR: {e}. Is another server running on port {PORT}?")
            return
            
        s.listen()
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()