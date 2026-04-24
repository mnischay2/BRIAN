import subprocess
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
python_path = os.path.join(current_dir, ".venv", "bin", "python")
files_list = [
    "central.py", "speaker.py", "ai_handler.py","transcribe.py", "mic.py","ui_client.py","session_mgr.py"
]
files_dir = os.path.join(current_dir, "scripts", "nodes")

for script in files_list:
    subprocess.Popen([
        "xterm", "-hold", "-e", f"{python_path} {script}"
    ])
